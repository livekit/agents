//! From sentence ends to the chunks a voice speaks, shared by the batch split and the stream.

use std::ops::Range;

use super::{Script, SentenceOptions, collapse_newlines, is_space, script, spoken_words};
use crate::lexer::{TagSpan, Vocabulary, tag_may_follow, tag_spans};
use lk_agents_abi::tts::types::TextUnit;

/// A text and what the sentence rules read of it: the text without its tags.
pub(super) struct View {
    pub orig: Vec<char>,
    pub clean: Vec<char>,
    tags: Vec<TagSpan>,
}

impl View {
    pub fn new(text: &str, markup: Option<&Vocabulary>) -> Self {
        let orig: Vec<char> = text.chars().collect();
        let tags = markup.map(|vocab| tag_spans(vocab, text)).unwrap_or_default();
        let mut clean = Vec::with_capacity(orig.len());
        let mut at = 0;
        for tag in &tags {
            clean.extend(&orig[at..tag.start]);
            at = tag.end;
        }
        clean.extend(&orig[at..]);
        View { orig, clean, tags }
    }

    /// The text position of a clean one; a tag there goes with what follows.
    pub fn to_orig(&self, clean: usize) -> usize {
        self.tags
            .iter()
            .fold(clean, |at, tag| if tag.start < at { at + tag.end - tag.start } else { at })
    }

    pub fn render(&self, span: Range<usize>, retain_format: bool) -> String {
        let raw: String = self.orig[span].iter().collect();
        if retain_format { raw } else { collapse_newlines(&raw) }
    }
}

/// Cuts the clean text at sentence ends and clause breaks, and batches the pieces into chunks.
/// Positions are clean characters; a stream shifts them as it drops what it emitted.
#[derive(Debug, Clone, Default)]
pub(super) struct Chunker {
    /// The last cut: a sentence end or a clause break.
    cut: usize,
    /// The next character to read as a clause mark.
    mark: usize,
    /// Where the chunk being batched starts, and its last cut so far.
    pub start: usize,
    held: Option<usize>,
}

impl Chunker {
    /// Cuts at the sentence `ends` (the scan's, in its order) and at the clause marks before
    /// `limit`, pushing the text span of each chunk completed.
    pub fn advance(
        &mut self,
        view: &View,
        opts: &SentenceOptions,
        ends: &[usize],
        limit: usize,
        out: &mut Vec<Range<usize>>,
    ) {
        for &end in ends {
            self.clauses(view, opts, end, out);
            if end >= self.cut && view.clean[self.cut..=end].iter().any(|&c| !is_space(c)) {
                self.cut_at(end + 1, view, opts, out);
            }
        }
        self.clauses(view, opts, limit, out);
    }

    /// The text is complete: what is held goes out, whatever its length.
    pub fn finish(&mut self, view: &View, opts: &SentenceOptions, out: &mut Vec<Range<usize>>) {
        let end = view.orig.len();
        let said = |from: usize| view.orig[from..].iter().any(|&c| !is_space(c));
        let from = view.to_orig(self.start);
        if let Some(held) = self.held
            && said(from)
            && over(view, opts, from..end)
        {
            out.push(from..view.to_orig(held));
            self.start = held;
        }
        let from = view.to_orig(self.start);
        if said(from) {
            out.push(from..end);
        }
    }

    pub fn shift(&mut self, by: usize) {
        self.cut -= by;
        self.mark = self.mark.saturating_sub(by);
        self.start -= by;
        self.held = self.held.map(|held| held - by);
    }

    /// A clause mark breaks a sentence once `clause_words` are said since the last cut, and a
    /// piece no rule cut is cut once it renders past the ceiling. Both read only the characters
    /// up to the one they decide on, so a stream cuts where the whole text does.
    fn clauses(
        &mut self,
        view: &View,
        opts: &SentenceOptions,
        until: usize,
        out: &mut Vec<Range<usize>>,
    ) {
        let ceiling = opts.max_token_len.unwrap_or(MAX_PIECE);
        while self.mark < until {
            let at = self.mark;
            self.mark += 1;
            if at < self.cut {
                continue;
            }
            if opts.unit == TextUnit::Words
                && at > self.cut
                && is_space(view.clean[at - 1])
                && !is_space(view.clean[at])
            {
                self.cut_at(at, view, opts, out);
            } else if let Some(words) = opts.clause_words
                && is_clause_break(&view.clean, at)
                && spoken_words(&view.clean[self.cut..=at].iter().collect::<String>()) >= words
            {
                self.cut_at(at + 1, view, opts, out);
            } else if over_ceiling(view, opts, self.cut..at + 1, ceiling) {
                self.cut_at(hard_cut(&view.clean, self.cut, at), view, opts, out);
            }
        }
    }

    fn cut_at(
        &mut self,
        cut: usize,
        view: &View,
        opts: &SentenceOptions,
        out: &mut Vec<Range<usize>>,
    ) {
        if cut <= self.cut {
            return;
        }
        self.cut = cut;
        let (from, to) = (view.to_orig(self.start), view.to_orig(cut));
        if let Some(held) = self.held
            && over(view, opts, from..to)
        {
            out.push(from..view.to_orig(held));
            self.start = held;
        }
        let from = view.to_orig(self.start);
        let clean: String = view.clean[self.start..cut].iter().collect();
        let open = opts.markup.as_ref().is_some_and(|vocab| {
            has_unclosed_tags(&view.orig[from..to].iter().collect::<String>(), vocab)
        });
        if spoken_words(&clean) >= opts.min_words.max(1) && !open {
            out.push(from..to);
            self.start = cut;
            self.held = None;
        } else {
            self.held = Some(cut);
        }
    }
}

/// Would this chunk pass `max_token_len`?
fn over(view: &View, opts: &SentenceOptions, span: Range<usize>) -> bool {
    opts.max_token_len
        .is_some_and(|max| view.render(span, opts.retain_format).chars().count() > max)
}

/// The ceiling of a piece no rule cut when `max_token_len` is unset: about a minute of speech.
const MAX_PIECE: usize = 1000;

/// Whether the clean span renders past `ceiling` characters (tags and all, as `over` measures).
fn over_ceiling(view: &View, opts: &SentenceOptions, clean: Range<usize>, ceiling: usize) -> bool {
    let span = view.to_orig(clean.start)..view.to_orig(clean.end);
    span.len() > ceiling && view.render(span, opts.retain_format).chars().count() > ceiling
}

/// Where to cut `text[from..to]`, which fits while `text[from..=to]` does not: after its last
/// clause mark, else at its last word boundary, else before `to`. A clean position is never
/// inside a tag.
fn hard_cut(text: &[char], from: usize, to: usize) -> usize {
    let cuts = (from + 1..=to).rev();
    let word = |b: usize| {
        let (before, after) = (text[b - 1], text[b]);
        (is_space(before) && !is_space(after))
            || !matches!(script(before), Script::Spaced)
            || !matches!(script(after), Script::Spaced)
    };
    cuts.clone()
        .find(|&b| is_clause_break(text, b - 1))
        .or_else(|| cuts.clone().find(|&b| word(b)))
        .unwrap_or(to)
}

/// `,` `;` `:` before a space (not `1,000` or `12:30`), or a fullwidth or ideographic one not
/// before a digit.
fn is_clause_break(text: &[char], at: usize) -> bool {
    let Some(&next) = text.get(at + 1) else { return false };
    match text[at] {
        ',' | ';' | ':' => is_space(next),
        '，' | '、' | '；' | '：' => !next.is_numeric(),
        _ => false,
    }
}

/// An open wrapper holds its text only while it opened within this many characters of the
/// end, so a wrapper the model never closes cannot hold the rest of a reply.
const MAX_WRAP: usize = 600;

/// Whether `text` ends in a tag still arriving, or leaves a recent tag open.
pub(super) fn has_unclosed_tags(text: &str, vocab: &Vocabulary) -> bool {
    let mut open = Vec::new();
    for tag in tag_spans(vocab, text).iter().filter(|tag| !tag.self_closing) {
        if tag.closing {
            open.pop();
        } else {
            open.push(tag.start);
        }
    }
    let len = text.chars().count();
    open.first().is_some_and(|start| len - start <= MAX_WRAP) || tag_may_follow(vocab, text)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: counting a self-closing tag as an open.
    #[test]
    fn a_self_closing_tag_leaves_nothing_open() {
        let all = Vocabulary::All;
        assert!(!has_unclosed_tags("hello <break/> world", &all));
        assert!(!has_unclosed_tags("hello <break time=\"1s\" /> world", &all));
        assert!(!has_unclosed_tags("<emphasis>hi</emphasis>", &all));
        assert!(has_unclosed_tags("<emphasis>hi", &all));
    }

    // Catches: a wrapper the model never closes holding every later sentence back.
    #[test]
    fn an_open_wrapper_holds_only_while_it_is_recent() {
        let v = Vocabulary::LiveKit;
        let open = "<expr type=\"prosody\" label=\"slow\">";
        assert!(has_unclosed_tags(&format!("{open}{}", "word ".repeat(100)), &v));
        assert!(!has_unclosed_tags(&format!("{open}{}", "word ".repeat(130)), &v));
    }

    // Catches: prose holding a sentence open (`a<b` is no tag under a voice's vocabulary).
    #[test]
    fn prose_angle_brackets_are_no_tags() {
        let v = Vocabulary::LiveKit;
        assert!(!has_unclosed_tags("if a<b then", &v));
        assert!(!has_unclosed_tags("3 < 5 and <b>bold", &v));
        assert!(has_unclosed_tags("say <ex", &v));
        assert!(has_unclosed_tags("say <expr type=\"prosody\" label=\"slow\">one.", &v));
    }

    // Catches: a tag at a cut going with the text before it.
    #[test]
    fn a_tag_at_a_cut_goes_with_what_follows() {
        let view = View::new("Hello. <break/> World.", Some(&Vocabulary::All));
        assert_eq!(view.clean.iter().collect::<String>(), "Hello.  World.");
        assert_eq!(view.to_orig(7), 7);
        assert_eq!(view.to_orig(8), 16);
        assert_eq!(view.to_orig(view.clean.len()), view.orig.len());
    }
}
