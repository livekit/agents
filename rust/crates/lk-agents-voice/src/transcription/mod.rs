pub mod effects;
pub mod filters;
pub mod machine;

use crate::transcription::filters::{MarkdownFilter, filter_emoji};
use lk_agents_core::tts::markup::{ExpressiveTag, TranscriptMarkupStripper, split_all_markup};
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

/// One piece of agent text, as the user sees it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Visible {
    /// The text as the model wrote it, markup included. The TTS is its only consumer.
    pub raw: String,
    /// The text with every markup tag removed. What the transcript legs and the item carry.
    pub text: String,
    /// The tags removed from `raw`, in order; the room publishes the leading one.
    pub tags: Vec<ExpressiveTag>,
}

impl Visible {
    /// Split a whole piece of text; for deltas use [`VisibleStream`].
    pub fn of(raw: &str) -> Visible {
        let (text, tags) = split_all_markup(raw);
        Visible { raw: raw.to_owned(), text, tags }
    }

    /// Nothing the user can see: a delta that was markup only, or a tag still being held.
    pub fn is_blank(&self) -> bool {
        self.text.is_empty()
    }
}

/// [`Visible`] over one segment's deltas: a tag split across deltas is held until it closes.
#[derive(Debug, Default)]
pub struct VisibleStream {
    stripper: TranscriptMarkupStripper,
    seen_tags: usize,
}

impl VisibleStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// One delta in; the text it made visible out (possibly none, while a tag is held).
    pub fn push(&mut self, raw: &str) -> Visible {
        let text = self.stripper.push(raw);
        Visible { raw: raw.to_owned(), text, tags: self.new_tags() }
    }

    /// The segment ended: whatever was held is visible now.
    pub fn flush(&mut self) -> Visible {
        let text = self.stripper.flush();
        Visible { raw: String::new(), text, tags: self.new_tags() }
    }

    fn new_tags(&mut self) -> Vec<ExpressiveTag> {
        let all = self.stripper.tags();
        let new = all[self.seen_tags.min(all.len())..].to_vec();
        self.seen_tags = all.len();
        new
    }
}

pub trait TextTransform: Send {
    /// Feed a chunk; return the pieces that are now final. May be empty.
    fn push(&mut self, chunk: &str) -> Vec<String>;
    fn flush(&mut self) -> Vec<String>;
}

use lk_agents_abi::voice::session_options::TextTransform as Builtin;

fn build(builtin: Builtin) -> Box<dyn TextTransform> {
    match builtin {
        Builtin::FilterMarkdown => Box::new(MarkdownTransform(MarkdownFilter::new())),
        Builtin::FilterEmoji => Box::new(EmojiTransform),
    }
}

/// [`filter_markdown`](crate::transcription::filters::MarkdownFilter) as a pipeline stage.
#[derive(Debug)]
pub struct MarkdownTransform(MarkdownFilter);

impl TextTransform for MarkdownTransform {
    fn push(&mut self, chunk: &str) -> Vec<String> {
        self.0.push(chunk)
    }
    fn flush(&mut self) -> Vec<String> {
        self.0.flush()
    }
}

/// [`filter_emoji`] as a pipeline stage. Stateless, so it never holds a chunk back.
#[derive(Debug, Default)]
pub struct EmojiTransform;

impl TextTransform for EmojiTransform {
    fn push(&mut self, chunk: &str) -> Vec<String> {
        vec![filter_emoji(chunk)]
    }
    fn flush(&mut self) -> Vec<String> {
        Vec::new()
    }
}

/// The text transform stages, in order.
pub struct Pipeline {
    stages: Vec<Box<dyn TextTransform>>,
}

impl Pipeline {
    /// Resolve and build. Fails on the first unknown name, before any text has been seen.
    pub fn new(builtins: &[Builtin]) -> Pipeline {
        Pipeline { stages: builtins.iter().copied().map(build).collect() }
    }
}

/// A pipeline is itself a transform, so one can be nested in another.
impl TextTransform for Pipeline {
    fn push(&mut self, chunk: &str) -> Vec<String> {
        let mut current = vec![chunk.to_owned()];
        for stage in &mut self.stages {
            let mut next = Vec::new();
            for piece in &current {
                next.extend(stage.push(piece));
            }
            current = next;
        }
        current
    }

    /// End of stream: each stage flushes after the pieces carried down from above.
    fn flush(&mut self) -> Vec<String> {
        let mut carried: Vec<String> = Vec::new();
        for stage in &mut self.stages {
            let mut next = Vec::new();
            for piece in &carried {
                next.extend(stage.push(piece));
            }
            next.extend(stage.flush());
            carried = next;
        }
        carried
    }
}
