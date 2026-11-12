//! The one reading of streamed model text every consumer shares: the LLM's reasoning filter, the
//! sentence splitter, the TTS, the transcript, the pacing and the history.
//!
//! The lexer turns deltas into typed [`Token`]s; each consumer reads them through a [`View`]. Four
//! optional [`Rules`]: reasoning delimiters, a markup [`Vocabulary`], markdown and emoji. Anything
//! no rule recognizes is prose, untouched: `a<b`, `<3`, `5 * 3`, a developer's own `<b>`.
//!
//! Every recognizer answers the text at a position with a match, a literal run, or "need more",
//! and says "need more" only while more text could change its answer, within a bounded window. So
//! the tokens never depend on how the stream was chunked, and nothing is held for long. When the
//! stream ends, interrupted or not, a tag or reasoning still open is dropped, never spoken or shown;
//! an unclosed markdown delimiter is literal.

mod emoji;
mod markdown;
mod markup;
mod model;
mod project;
mod sync;

pub use markup::{EXPR, Tag, Vocabulary};
pub use model::{ModelSyntax, Span, Turn, Turns};
pub use project::{Projector, View};
pub use sync::{Unit, Units};

/// How far an inline markdown construct may run, in bytes and on one line, before its opener is
/// taken literally.
pub const INLINE_WINDOW: usize = 160;

/// What one stream's lexer recognizes; everything else is prose.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Rules {
    /// What the model writes that is not for the person.
    pub model: Option<ModelSyntax>,
    /// The tags that are markup.
    pub tags: Option<Vocabulary>,
    pub markdown: bool,
    pub emoji: bool,
}

impl Rules {
    /// What the LLM stage reads: only the model's own syntax.
    pub fn model(syntax: ModelSyntax) -> Rules {
        Rules { model: Some(syntax), ..Rules::default() }
    }

    /// What a voice reads of a reply: markup in `vocab`, markdown and emoji.
    pub fn speech(vocab: Vocabulary) -> Rules {
        Rules { model: None, tags: Some(vocab), markdown: true, emoji: true }
    }
}

/// Where a prose token sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Context {
    /// Inside `~~...~~`: a correction, never spoken.
    pub struck: bool,
    /// Inside a fenced block: shown, never spoken.
    pub fenced: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: Kind,
    /// The text as the model wrote it.
    pub raw: String,
    pub ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    Prose,
    Tag(Tag),
    Emoji,
    Markdown(Md),
    /// Never shown, never spoken.
    Hidden(Hidden),
}

/// Why a token is hidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hidden {
    Reasoning,
    /// A tool call the model wrote as text: never run, and worth a warning.
    ToolCall,
    /// A chat-template control token, or a stray delimiter.
    Control,
    /// Turn structure: a role header, another role's turn the model invented, a channel other
    /// than `final`, or what follows the end of the turn.
    Channel,
    /// A botched, or cut, markup tag.
    Markup,
}

/// A markdown delimiter: the syntax, never the words it formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Md {
    EmphasisOpen,
    EmphasisClose,
    StrikeOpen,
    StrikeClose,
    CodeOpen,
    CodeClose,
    /// `[` or `![`.
    LinkOpen,
    /// `](url)`.
    LinkClose {
        url: String,
    },
    Heading,
    ListItem,
    Quote,
    Rule,
    FenceOpen,
    FenceClose,
}

/// A recognizer's answer at a position.
pub(crate) enum Scan {
    /// These tokens, over `len` bytes.
    Match { len: usize, tokens: Vec<Token> },
    /// `len` bytes of prose: the construct is not there.
    Literal(usize),
    /// More text could change the answer.
    NeedMore,
}

impl Scan {
    /// The bytes settled at the start of `s` and their tokens; `None` while more text is needed.
    fn settled(self, s: &str, ctx: Context) -> Option<(usize, Vec<Token>)> {
        match self {
            Scan::Match { len, tokens } => Some((len, tokens)),
            Scan::Literal(len) => {
                Some((len, vec![Token { kind: Kind::Prose, raw: s[..len].to_owned(), ctx }]))
            }
            Scan::NeedMore => None,
        }
    }
}

/// Tokens being assembled: neighbouring prose with one context merges.
#[derive(Default)]
struct Out {
    tokens: Vec<Token>,
}

impl Out {
    fn push(&mut self, token: Token) {
        if let Some(last) = self.tokens.last_mut()
            && last.ctx == token.ctx
            && matches!((&last.kind, &token.kind), (Kind::Prose, Kind::Prose))
        {
            last.raw.push_str(&token.raw);
            return;
        }
        if let (Some(last), Kind::Hidden(why)) = (self.tokens.last_mut(), &token.kind)
            && last.kind == Kind::Hidden(*why)
        {
            last.raw.push_str(&token.raw);
            return;
        }
        self.tokens.push(token);
    }
}

/// What the lexer knows of the text it already settled.
#[derive(Debug, Clone)]
struct State {
    model: model::ModelState,
    fence: Option<markdown::Fence>,
    /// The last character settled, for the intraword rules of emphasis and the line start.
    prev: Option<char>,
}

impl State {
    fn new(rules: &Rules) -> State {
        State {
            model: rules.model.as_ref().map(model::ModelState::new).unwrap_or_default(),
            fence: None,
            prev: None,
        }
    }

    /// Reasoning and markup are transparent: the text around them reads as if they were not there.
    fn settled(&mut self, token: &Token) {
        if matches!(token.kind, Kind::Hidden(_) | Kind::Tag(_)) {
            return;
        }
        self.prev = token.raw.chars().next_back().or(self.prev);
    }

    fn line_start(&self) -> bool {
        self.prev.is_none_or(|c| c == '\n')
    }
}

/// Incremental lexer over one stream of text.
#[derive(Debug, Clone)]
pub struct Lexer {
    rules: Rules,
    state: State,
    buf: String,
}

impl Lexer {
    pub fn new(rules: Rules) -> Lexer {
        Lexer { state: State::new(&rules), rules, buf: String::new() }
    }

    /// Feed a delta; the tokens it settled. A tail that could still become something is held.
    pub fn push(&mut self, text: &str) -> Vec<Token> {
        self.buf.push_str(text);
        let (tokens, held_from) = lex(&self.rules, &self.buf, &mut self.state, false);
        self.buf.drain(..held_from);
        tokens
    }

    /// The stream ended: the held tail settles, a tag or reasoning still open dropped.
    pub fn finish(&mut self) -> Vec<Token> {
        let buf = std::mem::take(&mut self.buf);
        let tokens = lex(&self.rules, &buf, &mut self.state, true).0;
        self.state = State::new(&self.rules);
        tokens
    }

    /// The bytes held back, waiting for more text.
    pub fn held(&self) -> usize {
        self.buf.len()
    }

    /// Lex one whole text.
    pub fn all(rules: Rules, text: &str) -> Vec<Token> {
        let mut lexer = Lexer::new(rules);
        let mut tokens = lexer.push(text);
        tokens.extend(lexer.finish());
        tokens
    }
}

/// Lex `text`; the tokens and where the held tail starts (`text.len()` when `at_end`).
fn lex(rules: &Rules, text: &str, state: &mut State, at_end: bool) -> (Vec<Token>, usize) {
    let mut out = Out::default();
    let mut p = 0;
    let settle = |out: &mut Out, state: &mut State, tokens: Vec<Token>| {
        for t in tokens {
            state.settled(&t);
            out.push(t);
        }
    };
    while p < text.len() {
        let s = &text[p..];
        if let Some(syntax) = &rules.model {
            match model::step(syntax, &mut state.model, s, at_end, &mut out) {
                model::Step::Hid(len) => {
                    p += len;
                    continue;
                }
                model::Step::Hold => return (out.tokens, p),
                model::Step::Pass => {}
            }
        }
        if rules.markdown && state.line_start() {
            let scan = match state.fence {
                Some(fence) => markdown::fence_line(fence, s, at_end),
                None => markdown::line_start(s, at_end),
            };
            match scan {
                Scan::Match { len, tokens } => {
                    for t in &tokens {
                        match t.kind {
                            Kind::Markdown(Md::FenceOpen) => {
                                state.fence = Some(markdown::Fence::of(&t.raw));
                            }
                            Kind::Markdown(Md::FenceClose) => state.fence = None,
                            _ => {}
                        }
                    }
                    settle(&mut out, state, tokens);
                    p += len;
                    continue;
                }
                Scan::NeedMore => return (out.tokens, p),
                Scan::Literal(_) => {}
            }
        }
        if state.fence.is_some() {
            // fenced content runs to its line end, settled as it arrives
            let end = s.find('\n').map_or(s.len(), |i| i + 1);
            let t = Token {
                kind: Kind::Prose,
                raw: s[..end].to_owned(),
                ctx: Context { fenced: true, ..Context::default() },
            };
            settle(&mut out, state, vec![t]);
            p += end;
            continue;
        }
        let ctx = Context::default();
        let Some((len, tokens)) = inline(rules, s, state.prev, ctx, at_end).settled(s, ctx) else {
            return (out.tokens, p);
        };
        settle(&mut out, state, tokens);
        p += len;
    }
    (out.tokens, text.len())
}

/// One inline construct at the start of `s`, or the prose up to the next place one could start.
fn inline(rules: &Rules, s: &str, prev: Option<char>, ctx: Context, at_end: bool) -> Scan {
    let scan = match s.chars().next() {
        Some(c) if may_start(rules, c) => match c {
            '<' => markup_scan(rules.tags.as_ref().expect("a tag may start"), s, ctx, at_end),
            '`' | '[' | '!' | '*' | '_' | '~' => markdown::inline(rules, s, prev, ctx, at_end),
            _ => emoji::run(s, ctx, at_end),
        },
        _ => Scan::Literal(0),
    };
    match scan {
        Scan::Literal(0) => Scan::Literal(prose_run(rules, s)),
        scan => scan,
    }
}

/// Whether an inline rule could start at `c`.
fn may_start(rules: &Rules, c: char) -> bool {
    match c {
        '<' => rules.tags.is_some(),
        '`' | '[' | '!' | '*' | '_' | '~' => rules.markdown,
        c if emoji::is_emoji(c) => rules.emoji,
        _ => false,
    }
}

/// The prose from the start of `s` up to the next character a rule could start at, at least one;
/// a newline ends a run, so the line after it starts fresh.
fn prose_run(rules: &Rules, s: &str) -> usize {
    let first = s.chars().next().map_or(0, char::len_utf8);
    let line_end = s.find('\n').map_or(s.len(), |i| i + 1);
    let model_starts: Vec<char> = rules.model.as_ref().map(ModelSyntax::starts).unwrap_or_default();
    s[first..line_end]
        .char_indices()
        .find(|(_, c)| model_starts.contains(c) || may_start(rules, *c))
        .map_or(line_end, |(i, _)| first + i)
}

/// A tag, a dropped fragment, or a literal `<`.
fn markup_scan(vocab: &Vocabulary, s: &str, ctx: Context, at_end: bool) -> Scan {
    let token = |kind, raw: &str| Token { kind, raw: raw.to_owned(), ctx };
    match markup::scan(vocab, s, at_end) {
        markup::Scan::Prose => Scan::Literal(1),
        markup::Scan::Tag { tag, end } => {
            Scan::Match { len: end, tokens: vec![token(Kind::Tag(tag), &s[..end])] }
        }
        markup::Scan::Malformed { end } => {
            Scan::Match { len: end, tokens: vec![token(Kind::Hidden(Hidden::Markup), &s[..end])] }
        }
        markup::Scan::Incomplete => Scan::NeedMore,
    }
}

/// Lex a complete inline span (an emphasis or link body): nothing in it waits for more text.
fn lex_inline(rules: &Rules, s: &str, ctx: Context) -> Vec<Token> {
    let mut out = Out::default();
    let mut prev = None;
    let mut p = 0;
    while p < s.len() {
        let (len, tokens) = inline(rules, &s[p..], prev, ctx, true)
            .settled(&s[p..], ctx)
            .expect("at the end every recognizer settles");
        for t in tokens {
            if !matches!(t.kind, Kind::Hidden(_) | Kind::Tag(_)) {
                prev = t.raw.chars().next_back().or(prev);
            }
            out.push(t);
        }
        p += len;
    }
    out.tokens
}

#[cfg(test)]
#[path = "latency_tests.rs"]
mod latency_tests;
#[cfg(test)]
#[path = "markdown_tests.rs"]
mod markdown_tests;
#[cfg(test)]
#[path = "markup_tests.rs"]
mod markup_tests;
#[cfg(test)]
#[path = "model_tests.rs"]
mod model_tests;
#[cfg(test)]
#[path = "properties_tests.rs"]
mod properties_tests;
#[cfg(test)]
#[path = "sentences_tests.rs"]
mod sentences_tests;
#[cfg(test)]
#[path = "sync_tests.rs"]
mod sync_tests;

/// The largest char boundary at or before `i`.
fn floor_char(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// A recognized tag in a text, as **character** offsets: what the sentence splitter keeps whole.
/// A botched tag counts too, as a self-closing one: it is dropped later, and never split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TagSpan {
    pub start: usize,
    pub end: usize,
    pub closing: bool,
    pub self_closing: bool,
}

/// The tags of `text` under `vocab`.
pub fn tag_spans(vocab: &Vocabulary, text: &str) -> Vec<TagSpan> {
    let mut out = Vec::new();
    let (mut i, mut chars) = (0, 0);
    while let Some(rel) = text[i..].find('<') {
        let at = i + rel;
        chars += text[i..at].chars().count();
        let (len, span) = match markup::scan(vocab, &text[at..], false) {
            markup::Scan::Tag { tag, end } => (end, Some((tag.closing, tag.self_closing))),
            markup::Scan::Malformed { end } => (end, Some((false, true))),
            markup::Scan::Prose | markup::Scan::Incomplete => (1, None),
        };
        let n = text[at..at + len].chars().count();
        if let Some((closing, self_closing)) = span {
            out.push(TagSpan { start: chars, end: chars + n, closing, self_closing });
        }
        chars += n;
        i = at + len;
    }
    out
}

/// The byte offset of the first `<` in `text` that could still become a tag under `vocab`.
pub fn pending_tag(vocab: &Vocabulary, text: &str) -> Option<usize> {
    let mut i = 0;
    while let Some(rel) = text[i..].find('<') {
        let at = i + rel;
        i = match markup::scan(vocab, &text[at..], false) {
            markup::Scan::Incomplete => return Some(at),
            markup::Scan::Tag { end, .. } | markup::Scan::Malformed { end } => at + end,
            markup::Scan::Prose => at + 1,
        };
    }
    None
}

/// Whether the end of `text` could still become a tag under `vocab`.
pub fn tag_may_follow(vocab: &Vocabulary, text: &str) -> bool {
    text.rfind('<').is_some_and(|at| {
        matches!(markup::scan(vocab, &text[at..], false), markup::Scan::Incomplete)
    })
}
