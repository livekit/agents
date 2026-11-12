//! Markup tags: the vocabulary, the tag grammar and the botched-tag recovery.

use super::floor_char;
use crate::markup::{Provider, ProviderExt};

/// The expressive dialect the guide teaches, recognized under every voice.
pub const EXPR: &str = "expr";

/// The longest tag the lexer holds back. A recognized tag still open past it was malformed.
pub(super) const MAX_TAG: usize = 256;

/// The tag names read as markup. `<expr>` is always one of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Vocabulary {
    /// LiveKit's own `<expr>` dialect, and nothing else.
    LiveKit,
    /// `<expr>` and these names, lowercase, longest first so `<short pause>` is not read as a
    /// shorter name.
    Names(Vec<String>),
    /// Any tag-shaped `<name ...>`: the developer said every tag is markup.
    All,
}

impl Vocabulary {
    /// `<expr>` and `provider`'s native tags: the default.
    pub fn of(provider: Option<Provider>) -> Vocabulary {
        match provider {
            Some(provider) => Vocabulary::names(provider.native_tags().iter().copied()),
            None => Vocabulary::LiveKit,
        }
    }

    /// `<expr>` and `names`.
    pub fn names<'a>(names: impl IntoIterator<Item = &'a str>) -> Vocabulary {
        Vocabulary::listing(names.into_iter().collect())
    }

    fn listing(names: Vec<&str>) -> Vocabulary {
        let mut all: Vec<String> = std::iter::once(EXPR)
            .chain(names)
            .map(|n| n.trim().to_ascii_lowercase())
            .filter(|n| !n.is_empty())
            .collect();
        all.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        all.dedup();
        if all.len() == 1 { Vocabulary::LiveKit } else { Vocabulary::Names(all) }
    }

    /// The listed names; `None` for [`Vocabulary::All`].
    pub fn listed(&self) -> Option<&[String]> {
        static LIVEKIT: std::sync::LazyLock<[String; 1]> =
            std::sync::LazyLock::new(|| [EXPR.to_owned()]);
        match self {
            Vocabulary::LiveKit => Some(&*LIVEKIT),
            Vocabulary::Names(names) => Some(names),
            Vocabulary::All => None,
        }
    }
}

/// A recognized tag, attribute values unquoted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tag {
    /// Lowercase, whatever case the model wrote.
    pub name: String,
    pub closing: bool,
    pub self_closing: bool,
    /// In written order, names lowercase.
    pub attrs: Vec<(String, String)>,
}

impl Tag {
    pub fn attr(&self, name: &str) -> Option<&str> {
        // a repeated key keeps the last occurrence
        self.attrs.iter().rev().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// The canonical spelling every later stage can rely on: lowercase name, double quotes, and
    /// no `<`, `>` or `"` inside a value.
    pub fn render(&self) -> String {
        if self.closing {
            return format!("</{}>", self.name);
        }
        let mut out = format!("<{}", self.name);
        for (k, v) in &self.attrs {
            let v: String =
                v.chars().map(|c| if matches!(c, '<' | '>' | '"') { ' ' } else { c }).collect();
            out.push_str(&format!(" {k}=\"{}\"", v.trim()));
        }
        out.push_str(if self.self_closing { "/>" } else { ">" });
        out
    }
}

/// How the text at a `<` reads.
pub(super) enum Scan {
    /// Prose: `<` is a character like any other.
    Prose,
    /// A tag from `<` up to `end` (exclusive).
    Tag { tag: Tag, end: usize },
    /// A recognized tag the model botched: the bytes up to `end` go, the rest is text.
    Malformed { end: usize },
    /// It could still become a tag: hold from `<`.
    Incomplete,
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

/// Read the text starting at a `<`. Only its first [`MAX_TAG`] bytes count: once that many have
/// arrived, the answer is the same however many more follow.
pub(super) fn scan(vocab: &Vocabulary, s: &str, at_end: bool) -> Scan {
    let full = s.len() >= MAX_TAG;
    let s = &s[..floor_char(s, s.len().min(MAX_TAG))];
    let b = s.as_bytes();
    let mut i = 1;
    let closing = b.get(i) == Some(&b'/');
    if closing {
        i += 1;
    }
    let Some((name, len)) = match_name(vocab, &s[i..]) else {
        return Scan::Prose;
    };
    // what arrived ends: a cut tag drops, a tag past its limit is a botched one
    let cut = |at: usize| match (full, at_end) {
        (true, _) => recover(s, at, true),
        (false, true) => Scan::Malformed { end: s.len() },
        (false, false) => Scan::Incomplete,
    };
    let Some(len) = len else {
        // a prefix of a name, at the end of what has arrived
        return if full { Scan::Prose } else { cut(s.len()) };
    };
    i += len;
    let mut tag = Tag { name, closing, self_closing: false, attrs: Vec::new() };
    loop {
        while b.get(i).copied().is_some_and(is_space) {
            i += 1;
        }
        match b.get(i) {
            None => return cut(i),
            Some(b'>') => return Scan::Tag { tag, end: i + 1 },
            Some(b'/') if !tag.closing => {
                let mut j = i + 1;
                while b.get(j).copied().is_some_and(is_space) {
                    j += 1;
                }
                return match b.get(j) {
                    None => cut(j),
                    Some(b'>') => {
                        tag.self_closing = true;
                        Scan::Tag { tag, end: j + 1 }
                    }
                    Some(_) => recover(s, j, at_end || full),
                };
            }
            Some(c) if !tag.closing && is_attr_char(*c) => {
                let start = i;
                while b.get(i).copied().is_some_and(is_attr_char) {
                    i += 1;
                }
                let key = s[start..i].to_ascii_lowercase();
                while b.get(i).copied().is_some_and(is_space) {
                    i += 1;
                }
                match b.get(i) {
                    None => return cut(start),
                    Some(b'=') => i += 1,
                    // a bare word: the model left the tag there, and the word is prose
                    Some(_) => return recover(s, start, at_end || full),
                }
                while b.get(i).copied().is_some_and(is_space) {
                    i += 1;
                }
                let quote = match b.get(i) {
                    None => return cut(i),
                    Some(q @ (b'"' | b'\'')) => *q,
                    Some(_) => return recover(s, i, at_end || full),
                };
                i += 1;
                let value_start = i;
                loop {
                    match b.get(i) {
                        None if full => return unterminated_value(s, value_start),
                        // cut inside the value: nothing after `<` was meant to be heard
                        None => return cut(i),
                        // `label='it's fine'`: a quote before a letter is an apostrophe
                        Some(c)
                            if *c == quote
                                && !b
                                    .get(i + 1)
                                    .copied()
                                    .is_some_and(|n| n.is_ascii_alphanumeric()) =>
                        {
                            break;
                        }
                        Some(b'<' | b'\n') => return unterminated_value(s, value_start),
                        Some(_) => i += 1,
                    }
                }
                tag.attrs.push((key, s[value_start..i].to_owned()));
                i += 1;
            }
            Some(_) => return recover(s, i, at_end || full),
        }
    }
}

fn is_attr_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'-'
}

fn is_boundary(c: u8) -> bool {
    is_space(c) || c == b'/' || c == b'>'
}

/// The name at the start of `s`: `Some((name, Some(len)))` for a whole name followed by a
/// boundary, `Some((name, None))` when what arrived is still a prefix of one, `None` otherwise.
fn match_name(vocab: &Vocabulary, s: &str) -> Option<(String, Option<usize>)> {
    let names = match vocab.listed() {
        Some(names) => names,
        None => {
            let b = s.as_bytes();
            if b.first().is_some_and(|c| !c.is_ascii_alphabetic()) {
                return None;
            }
            let n =
                b.iter().take_while(|c| c.is_ascii_alphanumeric() || b"_-:".contains(c)).count();
            let name = s[..n].to_ascii_lowercase();
            return match b.get(n) {
                None => Some((name, None)),
                Some(c) if is_boundary(*c) => Some((name, Some(n))),
                Some(_) => None,
            };
        }
    };
    let mut prefix_of: Option<&String> = None;
    for name in names {
        let n = name.len();
        if s.len() >= n {
            if s.as_bytes()[..n].eq_ignore_ascii_case(name.as_bytes()) {
                match s.as_bytes().get(n) {
                    None => prefix_of = prefix_of.or(Some(name)),
                    Some(c) if is_boundary(*c) => return Some((name.clone(), Some(n))),
                    Some(_) => {}
                }
            }
        } else if name.as_bytes()[..s.len()].eq_ignore_ascii_case(s.as_bytes()) {
            prefix_of = prefix_of.or(Some(name));
        }
    }
    prefix_of.map(|name| (name.clone(), None))
}

/// A recognized tag broke its grammar at `at`. When a `>` closes it before any other `<`, the model
/// meant that `>`: the whole tag goes. Otherwise only up to `at`, so the prose after a botched
/// marker survives. `settled`: no more of the tag can arrive.
fn recover(s: &str, at: usize, settled: bool) -> Scan {
    match s[1..].find(['<', '>']).map(|p| p + 1) {
        Some(gt) if s.as_bytes()[gt] == b'>' => Scan::Malformed { end: gt + 1 },
        // its `>` may still be on its way
        None if !settled => Scan::Incomplete,
        _ => Scan::Malformed { end: at },
    }
}

/// A quoted value never closed: the model forgot the quote. A `>` inside it ends the tag.
fn unterminated_value(s: &str, value_start: usize) -> Scan {
    let window = &s[value_start..];
    let stop = window.find(['<', '\n']).unwrap_or(window.len());
    match window[..stop].find('>') {
        Some(gt) => Scan::Malformed { end: value_start + gt + 1 },
        // the value runs to where it broke off: all of it was the marker's
        None => Scan::Malformed { end: value_start + stop },
    }
}
