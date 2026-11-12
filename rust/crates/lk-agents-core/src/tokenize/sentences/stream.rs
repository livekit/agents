use super::chunks::{Chunker, View};
use super::lexer::{Caches, Text};
use super::{SentenceOptions, rules};
use crate::lexer::pending_tag;

/// [`SentenceSplitter::sentences`](super::SentenceSplitter::sentences) as the text streams in:
/// a chunk goes out once no text to come could change it.
pub struct SentenceStream {
    caches: Caches<'static>,
    opts: SentenceOptions,
    /// The text after the last chunk emitted.
    pending: String,
    /// Where the next scan starts in the clean pending text, and whether that is the text's start.
    scan_from: usize,
    at_start: bool,
    chunker: Chunker,
}

impl SentenceStream {
    pub fn new(opts: SentenceOptions) -> Self {
        SentenceStream {
            caches: Caches::new(rules::rule_sets()),
            opts,
            pending: String::new(),
            scan_from: 0,
            at_start: true,
            chunker: Chunker::default(),
        }
    }

    /// The chunks `text` completes.
    pub fn push(&mut self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        self.pending.push_str(text);
        self.step(false)
    }

    /// Ends the text: the rest goes out, and what follows starts a new text.
    pub fn flush(&mut self) -> Vec<String> {
        let out = self.step(true);
        self.pending.clear();
        (self.scan_from, self.at_start, self.chunker) = (0, true, Chunker::default());
        out
    }

    fn step(&mut self, closed: bool) -> Vec<String> {
        // a tag still arriving is not read until it is whole
        let known = match &self.opts.markup {
            Some(vocab) if !closed => pending_tag(vocab, &self.pending),
            _ => None,
        };
        let known = known.unwrap_or(self.pending.len());
        let view = View::new(&self.pending[..known], self.opts.markup.as_ref());
        let from = self.scan_from;
        let text = Text { chars: &view.clean[from..], at_start: self.at_start, open: !closed };
        let (ends, stop) = self.caches.decided_ends(text);
        if let Some(scanned) = stop.checked_sub(usize::from(self.at_start)) {
            self.scan_from += scanned;
            self.at_start = false;
        }
        let ends: Vec<usize> = ends.into_iter().map(|end| end + from).collect();
        // a clause mark is read once the character after it is known
        let limit = if closed {
            view.clean.len()
        } else {
            self.scan_from.min(view.clean.len().saturating_sub(1))
        };
        let mut chunks = Vec::new();
        self.chunker.advance(&view, &self.opts, &ends, limit, &mut chunks);
        if closed {
            self.chunker.finish(&view, &self.opts, &mut chunks);
        }
        let out = chunks.into_iter().map(|c| view.render(c, self.opts.retain_format)).collect();
        let emitted = self.chunker.start;
        let bytes: usize = view.orig[..view.to_orig(emitted)].iter().map(|c| c.len_utf8()).sum();
        self.pending.drain(..bytes);
        self.scan_from -= emitted;
        self.chunker.shift(emitted);
        out
    }
}

impl lk_agents_abi::SentenceSplitter for SentenceStream {
    fn push(&mut self, text: &str) -> Vec<String> {
        SentenceStream::push(self, text)
    }

    fn flush(&mut self) -> Vec<String> {
        SentenceStream::flush(self)
    }
}

impl lk_agents_abi::tokenize::bindings::SpeechSentenceStream for SentenceStream {
    fn push(&mut self, text: String) -> Vec<String> {
        SentenceStream::push(self, &text)
    }

    fn flush(&mut self) -> Vec<String> {
        SentenceStream::flush(self)
    }
}
