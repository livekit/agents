pub use crate::transcription::filters::{MarkdownFilter, filter_emoji};
pub use crate::transcription::synchronizer::{
    Hyphenator, Liang, PaceEffect, PaceEvent, PaceState, PaceWarning, Pacer, PacerOptions,
    SEGMENT_ACLOSE_TIMEOUT, STANDARD_SPEECH_RATE, SyncConfig, TranscriptSynchronizer,
    hyphenate_word,
};
pub use crate::transcription::text_transforms::{Pipeline, Replace, TransformSpec, UnknownTransform};
use lk_agents_abi::tts::markup::{ExpressiveTag, TranscriptMarkupStripper, split_all_markup};
use std::time::Duration;

pub mod filters;

pub mod synchronizer;

pub mod text_transforms;

/// What was heard of one segment: the words, where the playhead stopped, whether it played out.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Heard {
    pub text: String,
    pub position: Duration,
    pub complete: bool,
    /// Zero for a segment with no audio.
    pub frames: u64,
}

impl Heard {
    pub fn complete(text: impl Into<String>, position: Duration, frames: u64) -> Heard {
        Heard { text: text.into(), position, complete: true, frames }
    }

    /// A segment cut at `position`; `text` is what was heard up to there.
    pub fn cut(text: impl Into<String>, position: Duration, frames: u64) -> Heard {
        Heard { text: text.into(), position, complete: false, frames }
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
