pub mod effects;
pub mod machine;

use lk_agents_abi::TranscriptChunk;
use lk_agents_abi::tts::markup::Markup;
use lk_agents_abi::voice::session_options::TextTransform;
use lk_agents_core::lexer::{Lexer, Projector, Rules, Sync, Unit, View, Vocabulary};
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

/// A plain string a host hands the transcript chain (a `transcription_node` yielding strings, a
/// custom output): it never went through a voice's reading, so its markup (any provider's, the
/// voice unknown here) is taken out and its delivery lifted onto the chunk. Per segment.
#[derive(Debug, Default)]
pub struct HostText {
    stripper: lk_agents_core::tts::markup::TranscriptMarkupStripper,
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

/// How a voice reads its replies: the markup its TTS speaks, and which `text_transforms` are on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechViews {
    pub vocab: Vocabulary,
    pub filter_markdown: bool,
    pub filter_emoji: bool,
}

impl Default for SpeechViews {
    /// No TTS dialect, both transforms on: the session's defaults.
    fn default() -> Self {
        SpeechViews { vocab: Vocabulary::LiveKit, filter_markdown: true, filter_emoji: true }
    }
}

impl SpeechViews {
    pub fn new(markup: Markup, transforms: &[TextTransform]) -> SpeechViews {
        SpeechViews {
            vocab: Vocabulary::of(markup.0),
            filter_markdown: transforms.contains(&TextTransform::FilterMarkdown),
            filter_emoji: transforms.contains(&TextTransform::FilterEmoji),
        }
    }

    fn said(&self) -> View {
        View::Synthesis { keep_markdown: !self.filter_markdown, keep_emoji: !self.filter_emoji }
    }

    fn spoken(&self) -> View {
        View::Spoken { keep_markdown: !self.filter_markdown, keep_emoji: !self.filter_emoji }
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
    /// A whole piece of text; for deltas use [`SegmentText`].
    pub fn of(text: &str, views: &SpeechViews) -> Visible {
        let mut segment = SegmentText::new(views.clone());
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
    units: Sync,
    /// A delivery met since the last chunk: it goes on the next one.
    expression: Option<String>,
}

impl SegmentText {
    pub fn new(views: SpeechViews) -> SegmentText {
        SegmentText {
            lexer: Lexer::new(Rules::speech(views.vocab.clone())),
            said: Projector::new(views.said()),
            plain: Projector::new(View::Plain),
            units: Sync::new(views.spoken(), View::Display { keep_emoji: true }),
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
                        tags.iter().rev().find_map(lk_agents_core::tts::markup::delivery)
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
