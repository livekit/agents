//! Markdown: line markers at a line start, fences, and the inline delimiters. Each emits its
//! delimiters as tokens around a body lexed like any other text, so markup, code or emoji inside
//! emphasis keep their own meaning.

use super::{
    Context, INLINE_WINDOW, Kind, Md, Rules, Scan, Token, floor_char, lex_inline, markup_scan,
};
use crate::tokenize::is_py_space;

fn md(kind: Md, raw: &str, ctx: Context) -> Token {
    Token { kind: Kind::Markdown(kind), raw: raw.to_owned(), ctx }
}

fn is_blank(b: u8) -> bool {
    b == b' ' || b == b'\t'
}

/// `read` the current line of `s` with its newline, told whether the line is whole. A line is
/// decided within the inline window: a longer one reads as if it went on past the window, and what
/// would still wait there is literal.
fn at_line(s: &str, at_end: bool, read: impl FnOnce(&str, bool) -> Scan) -> Scan {
    let (line, whole) = match s.find('\n') {
        Some(i) => (&s[..=i], true),
        None => (s, at_end),
    };
    if line.len() <= INLINE_WINDOW {
        return read(line, whole);
    }
    match read(&line[..floor_char(line, INLINE_WINDOW)], false) {
        Scan::NeedMore => Scan::Literal(0),
        scan => scan,
    }
}

/// Up to three leading spaces, as a block marker allows.
fn indent3(b: &[u8]) -> usize {
    b.iter().take(3).take_while(|c| **c == b' ').count()
}

/// An open fenced block: its fence character and length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Fence {
    ch: u8,
    len: usize,
}

impl Fence {
    /// The fence a `FenceOpen` token opened.
    pub(super) fn of(raw: &str) -> Fence {
        let t = raw.trim_start_matches(' ');
        let ch = *t.as_bytes().first().expect("a fence opener starts with its fence");
        Fence { ch, len: t.bytes().take_while(|b| *b == ch).count() }
    }
}

enum Maybe {
    Yes,
    No,
    /// Not decided until more of the line arrives.
    Open,
}

/// `^ {0,3}([-*_])([ \t]*\1){2,}[ \t]*$`: the whole line, so it waits for the line end.
fn rule(line: &str, whole: bool) -> Maybe {
    let b = line.trim_end_matches(['\n', '\r']).as_bytes();
    let i = indent3(b);
    let Some(&marker) = b.get(i) else {
        return if whole { Maybe::No } else { Maybe::Open };
    };
    if !matches!(marker, b'-' | b'*' | b'_') {
        return Maybe::No;
    }
    let mut count = 0;
    for &c in &b[i..] {
        if c == marker {
            count += 1;
        } else if !is_blank(c) {
            return Maybe::No;
        }
    }
    match (whole, count >= 3) {
        (true, true) => Maybe::Yes,
        (true, false) => Maybe::No,
        (false, _) => Maybe::Open,
    }
}

/// The markers a line may start with: a rule, a fence, a heading, a list item, a quote.
pub(super) fn line_start(s: &str, at_end: bool) -> Scan {
    at_line(s, at_end, line_marker)
}

fn line_marker(line: &str, whole: bool) -> Scan {
    match rule(line, whole) {
        Maybe::Yes => {
            return Scan::Match {
                len: line.len(),
                tokens: vec![md(Md::Rule, line, Context::default())],
            };
        }
        Maybe::Open => return Scan::NeedMore,
        Maybe::No => {}
    }
    let b = line.as_bytes();
    // a fence: ` {0,3}` and three or more of one fence character, then its info string
    let i = indent3(b);
    if let Some(&ch @ (b'`' | b'~')) = b.get(i) {
        let n = b[i..].iter().take_while(|c| **c == ch).count();
        if i + n == b.len() && !whole {
            return Scan::NeedMore;
        }
        let info = &line[i + n..];
        if n >= 3 && !(ch == b'`' && info.contains('`')) {
            if !whole {
                return Scan::NeedMore;
            }
            let token = md(Md::FenceOpen, line, Context::default());
            return Scan::Match { len: line.len(), tokens: vec![token] };
        }
    }
    // `#{1,6}[ \t]+`, no indent
    let hashes = b.iter().take_while(|c| **c == b'#').count();
    if hashes > 0 {
        return marker(line, whole, hashes, hashes <= 6, Md::Heading);
    }
    // `\s*[-+*][ \t]+` and `\s*>[ \t]+`
    let indent = b.iter().take_while(|c| is_blank(**c)).count();
    match b.get(indent) {
        None if !whole => Scan::NeedMore,
        Some(b'-' | b'+' | b'*') => marker(line, whole, indent + 1, true, Md::ListItem),
        Some(b'>') => marker(line, whole, indent + 1, true, Md::Quote),
        _ => Scan::Literal(0),
    }
}

/// A block marker of `len` bytes, valid only when followed by blanks, which it takes along.
fn marker(line: &str, whole: bool, len: usize, valid: bool, kind: Md) -> Scan {
    let b = line.as_bytes();
    let blanks = b[len..].iter().take_while(|c| is_blank(**c)).count();
    if len + blanks == b.len() && !whole {
        return Scan::NeedMore;
    }
    if !valid || blanks == 0 {
        return Scan::Literal(0);
    }
    let raw = &line[..len + blanks];
    Scan::Match { len: raw.len(), tokens: vec![md(kind, raw, Context::default())] }
}

/// A line inside a fenced block: its closing fence, or content.
pub(super) fn fence_line(fence: Fence, s: &str, at_end: bool) -> Scan {
    at_line(s, at_end, |line, whole| fence_close(fence, line, whole))
}

fn fence_close(fence: Fence, line: &str, whole: bool) -> Scan {
    let b = line.trim_end_matches(['\n', '\r']).as_bytes();
    let i = indent3(b);
    let n = b[i..].iter().take_while(|c| **c == fence.ch).count();
    let rest_blank = b[i + n..].iter().all(|c| is_blank(*c));
    let could_close = rest_blank && (n >= fence.len || i + n == b.len());
    match (could_close, whole) {
        (true, false) => Scan::NeedMore,
        (true, true) if n >= fence.len => Scan::Match {
            len: line.len(),
            tokens: vec![md(Md::FenceClose, line, Context::default())],
        },
        _ => Scan::Literal(0),
    }
}

/// An inline construct starting with one of `` ` `` `[` `!` `*` `_` `~`.
pub(super) fn inline(
    rules: &Rules,
    s: &str,
    prev: Option<char>,
    ctx: Context,
    at_end: bool,
) -> Scan {
    match s.as_bytes()[0] {
        b'`' => code_span(s, prev, ctx, at_end),
        b'[' => link(rules, s, 1, ctx, at_end),
        b'!' => match s.as_bytes().get(1) {
            None if !at_end => Scan::NeedMore,
            Some(b'[') => match link(rules, s, 2, ctx, at_end) {
                Scan::Literal(_) => Scan::Literal(1),
                scan => scan,
            },
            _ => Scan::Literal(1),
        },
        b'*' => emphasis(rules, s, prev, ctx, at_end, is_intraword),
        b'_' => emphasis(rules, s, prev, ctx, at_end, is_word),
        // `~`, the last the caller lets through
        _ => strike(rules, s, prev, ctx, at_end),
    }
}

/// `` `code` ``: a run of backticks closed by a run of the same length, on the line. Its content is
/// literal. A run is read from its start: the rest of one cut at the window is literal.
fn code_span(s: &str, prev: Option<char>, ctx: Context, at_end: bool) -> Scan {
    let b = s.as_bytes();
    let n = b.iter().take_while(|c| **c == b'`').count();
    if prev == Some('`') {
        return Scan::Literal(n);
    }
    let mut i = n;
    loop {
        if i > INLINE_WINDOW {
            return Scan::Literal(n);
        }
        match b.get(i) {
            None => return if at_end { Scan::Literal(n) } else { Scan::NeedMore },
            Some(b'\n') => return Scan::Literal(n),
            Some(b'`') => {
                let m = b[i..].iter().take_while(|c| **c == b'`').count();
                if i + m == b.len() && !at_end && m <= n {
                    return Scan::NeedMore;
                }
                if m == n {
                    let code = Context { code: true, ..ctx };
                    let tokens = vec![
                        md(Md::CodeOpen, &s[..n], ctx),
                        Token { kind: Kind::Prose, raw: s[n..i].to_owned(), ctx: code },
                        md(Md::CodeClose, &s[i..i + n], ctx),
                    ];
                    return Scan::Match { len: i + n, tokens };
                }
                i += m;
            }
            Some(_) => i += s[i..].chars().next().map_or(1, char::len_utf8),
        }
    }
}

/// `[text](url)` (or `![alt](url)`, `open` = 2): the text is lexed, the url only shown.
fn link(rules: &Rules, s: &str, open: usize, ctx: Context, at_end: bool) -> Scan {
    let until = |from: usize, stop: char| -> Result<usize, Scan> {
        let mut i = from;
        loop {
            if i > INLINE_WINDOW {
                return Err(Scan::Literal(1));
            }
            match s[i..].chars().next() {
                None => return Err(if at_end { Scan::Literal(1) } else { Scan::NeedMore }),
                Some('\n') => return Err(Scan::Literal(1)),
                Some(c) if c == stop => return Ok(i),
                Some(c) => i += c.len_utf8(),
            }
        }
    };
    let text_end = match until(open, ']') {
        Ok(i) => i,
        Err(scan) => return scan,
    };
    match s.as_bytes().get(text_end + 1) {
        None if !at_end => return Scan::NeedMore,
        Some(b'(') => {}
        _ => return Scan::Literal(1),
    }
    let close = match until(text_end + 2, ')') {
        Ok(i) => i,
        Err(scan) => return scan,
    };
    let mut tokens = vec![md(Md::LinkOpen, &s[..open], ctx)];
    tokens.extend(lex_inline(rules, &s[open..text_end], ctx));
    let url = s[text_end + 2..close].to_owned();
    tokens.push(md(Md::LinkClose { url }, &s[text_end..=close], ctx));
    Scan::Match { len: close + 1, tokens }
}

/// The end of an inline body: the first `stop` byte on the line, stepping over markup and code
/// spans whole. `Err` is the opener's answer when there is none.
fn body_end(
    rules: &Rules,
    s: &str,
    from: usize,
    stop: u8,
    ctx: Context,
    at_end: bool,
    literal: usize,
) -> Result<usize, Scan> {
    let mut i = from;
    loop {
        if i > INLINE_WINDOW {
            return Err(Scan::Literal(literal));
        }
        let rest = &s[i..];
        let Some(c) = rest.chars().next() else {
            return Err(if at_end { Scan::Literal(literal) } else { Scan::NeedMore });
        };
        let step = match c {
            '\n' => return Err(Scan::Literal(literal)),
            c if c as u32 == u32::from(stop) => return Ok(i),
            '<' if rules.tags.is_some() => {
                markup_scan(rules.tags.as_ref().expect("checked above"), rest, ctx, at_end)
            }
            '`' => code_span(rest, None, ctx, at_end),
            c => Scan::Literal(c.len_utf8()),
        };
        i += match step {
            Scan::Match { len, .. } | Scan::Literal(len) => len,
            Scan::NeedMore => return Err(Scan::NeedMore),
        };
    }
}

/// `*x*`, `**x**`, `***x***` and the `_` twins: a run of one to three, no blank inside either
/// delimiter, no delimiter inside the body, and no word character against the outside of either
/// run (`2*3*4`, `snake_case`). The asterisk boundary leaves flush scripts (CJK, Thai) alone.
fn emphasis(
    rules: &Rules,
    s: &str,
    prev: Option<char>,
    ctx: Context,
    at_end: bool,
    boundary: fn(char) -> bool,
) -> Scan {
    let b = s.as_bytes();
    let d = b[0];
    let run = b.iter().take_while(|c| **c == d).count();
    if run > 3 || prev.is_some_and(|p| boundary(p) || p == d as char) {
        return Scan::Literal(run);
    }
    if run == b.len() && !at_end {
        return Scan::NeedMore;
    }
    match s[run..].chars().next() {
        None => return Scan::Literal(run),
        Some(c) if is_py_space(c) => return Scan::Literal(run),
        Some(_) => {}
    }
    let end = match body_end(rules, s, run, d, ctx, at_end, run) {
        Ok(end) => end,
        Err(scan) => return scan,
    };
    if s[..end].chars().next_back().is_some_and(is_py_space) {
        return Scan::Literal(run);
    }
    let close = b[end..].iter().take_while(|c| **c == d).count();
    if end + close == b.len() && !at_end && close <= run {
        return Scan::NeedMore;
    }
    if close != run {
        return Scan::Literal(run);
    }
    let close_end = end + run;
    if s[close_end..].chars().next().is_some_and(boundary) {
        return Scan::Literal(run);
    }
    let mut tokens = vec![md(Md::EmphasisOpen, &s[..run], ctx)];
    tokens.extend(lex_inline(rules, &s[run..end], ctx));
    tokens.push(md(Md::EmphasisClose, &s[end..close_end], ctx));
    Scan::Match { len: close_end, tokens }
}

/// `~~x~~`: struck text, never spoken.
fn strike(rules: &Rules, s: &str, prev: Option<char>, ctx: Context, at_end: bool) -> Scan {
    let b = s.as_bytes();
    let run = b.iter().take_while(|c| **c == b'~').count();
    if run > 2 || prev == Some('~') {
        return Scan::Literal(run);
    }
    if run == b.len() && !at_end {
        return Scan::NeedMore;
    }
    if run != 2 {
        return Scan::Literal(run);
    }
    match s[2..].chars().next() {
        None => return Scan::Literal(2),
        Some(c) if is_py_space(c) => return Scan::Literal(2),
        Some(_) => {}
    }
    let end = match body_end(rules, s, 2, b'~', ctx, at_end, 2) {
        Ok(end) => end,
        Err(scan) => return scan,
    };
    if s[..end].chars().next_back().is_some_and(is_py_space) {
        return Scan::Literal(2);
    }
    let close = b[end..].iter().take_while(|c| **c == b'~').count();
    if end + close == b.len() && !at_end && close <= 2 {
        return Scan::NeedMore;
    }
    if close != 2 {
        return Scan::Literal(2);
    }
    let struck = Context { struck: true, ..ctx };
    let mut tokens = vec![md(Md::StrikeOpen, &s[..2], ctx)];
    tokens.extend(lex_inline(rules, &s[2..end], struck));
    tokens.push(md(Md::StrikeClose, &s[end..end + 2], ctx));
    Scan::Match { len: end + 2, tokens }
}

/// Scripts that put an emphasis delimiter flush against a word (CJK, kana, Thai, Korean).
fn is_flush_emphasis_script(c: char) -> bool {
    matches!(c,
        '\u{0e00}'..='\u{0e7f}'   // thai
        | '\u{1100}'..='\u{11ff}' // hangul jamo
        | '\u{3040}'..='\u{30ff}' // hiragana, katakana
        | '\u{3130}'..='\u{318f}' // hangul compatibility jamo
        | '\u{3400}'..='\u{4dbf}' // cjk unified ideographs extension a
        | '\u{4e00}'..='\u{9fff}' // cjk unified ideographs
        | '\u{ac00}'..='\u{d7af}' // hangul syllables
        | '\u{f900}'..='\u{faff}' // cjk compatibility ideographs
        | '\u{ff66}'..='\u{ff9d}' // halfwidth katakana
    )
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// What makes an asterisk intraword, so `2*3*4` is not emphasis.
fn is_intraword(c: char) -> bool {
    is_word(c) && !is_flush_emphasis_script(c)
}
