//! `tts/markup_utils.py`: the XML tag stripper, the bracket rewrite, and the whitespace rule every
//! removal goes through. The patterns are the reference's, transcribed for `fancy_regex`: CPython's
//! `\s` is written `[\s\x1c-\x1f]` (`str.isspace` also counts U+001C..U+001F), its `\w` is
//! `[\p{L}\p{N}_]` (`str.isalnum() or "_"`), and `\b` is spelled out from that `\w`.

use std::borrow::Cow;
use std::sync::LazyLock;

use fancy_regex::{Regex, RegexBuilder};

pub(crate) type Captures<'t> = fancy_regex::Captures<'t, str>;

use crate::tts::markup::provider::ALL_MARKUP_TAGS;
use crate::tokenize::is_py_space;

/// CPython's `\b` for `str` patterns.
pub(crate) const WORD_BOUNDARY: &str =
    r"(?:(?<=[\p{L}\p{N}_])(?![\p{L}\p{N}_])|(?<![\p{L}\p{N}_])(?=[\p{L}\p{N}_]))";

/// `LEADING_WS`: horizontal whitespace immediately before a tag, captured as `pre` so
/// [`dedup_removal_space`] can decide whether to keep it. Newlines are excluded.
pub(crate) const LEADING_WS: &str = r"(?P<pre>[[\s\x1c-\x1f]&&[^\r\n]]*)";

/// A reference pattern. No backtrack limit: CPython has none, so a long input is slow, never an
/// error (`replace_all` would panic on one).
pub(crate) fn compile(pattern: &str) -> Regex {
    RegexBuilder::new(pattern).backtrack_limit(usize::MAX).build().expect("markup pattern")
}

/// `pattern.sub(repl, text)`.
pub(crate) fn sub(re: &Regex, text: &str, repl: impl FnMut(&Captures) -> String) -> String {
    re.replace_all(text, repl).into_owned()
}

pub(crate) use crate::tokenize::py_strip;

pub(crate) static EXPRESSION_RE: LazyLock<Regex> = LazyLock::new(|| {
    compile(r#"<expression[\s\x1c-\x1f]+value="([^"]*)"(?:[\s\x1c-\x1f]*/>|>(?:.*?)</expression>)"#)
});
static SOUND_RE: LazyLock<Regex> = LazyLock::new(|| {
    compile(r#"<sound[\s\x1c-\x1f]+value="([^"]*)"(?:[\s\x1c-\x1f]*/>|>(?:.*?)</sound>)"#)
});
static VALUE_ATTR_RE: LazyLock<Regex> = LazyLock::new(|| {
    compile(&format!(r#"{WORD_BOUNDARY}[\p{{L}}\p{{N}}_-]+[\s\x1c-\x1f]*=[\s\x1c-\x1f]*"([^"]*)""#))
});
/// The `extract_and_strip` pattern over [`ALL_MARKUP_TAGS`], the one every transcript uses.
static ALL_MARKUP_RE: LazyLock<Regex> = LazyLock::new(|| strip_pattern(&ALL_MARKUP_TAGS));

/// `_dedup_removal_space` (#6717): the replacement for a match whose pattern captured `pre`, where
/// `kept` is what survives (`""` when the tag vanishes). When nothing survives and whitespace
/// follows, `pre` is dropped; before a tag at the very end of `text` it is kept.
pub(crate) fn dedup_removal_space(m: &Captures, text: &str, kept: &str) -> String {
    let pre = m.name("pre").map_or("", |p| p.as_str());
    if !kept.is_empty() {
        return format!("{pre}{kept}");
    }
    let end = m.get(0).expect("group 0").end();
    match text[end..].chars().next() {
        Some(c) if !pre.is_empty() && is_py_space(c) => String::new(),
        _ => pre.to_string(),
    }
}

/// `<expression value="X"/>` and `<sound value="X"/>` (wrapping form included) -> `[X]`.
#[crate::host(export(pid = false))]
pub fn convert_expression_tags(text: &str) -> String {
    let text = sub(&EXPRESSION_RE, text, |m| format!("[{}]", &m[1]));
    sub(&SOUND_RE, &text, |m| format!("[{}]", &m[1]))
}

fn strip_pattern(xml_tags: &[&str]) -> Regex {
    let tags = xml_tags.iter().map(|t| regex::escape(t)).collect::<Vec<_>>().join("|");
    compile(&format!(
        r"(?s){LEADING_WS}(?:<(?P<tag>{tags}){WORD_BOUNDARY}(?P<attrs>[^>]*?)[\s\x1c-\x1f]*/?[\s\x1c-\x1f]*>(?:(?P<inner>.*?)</(?P=tag)[\s\x1c-\x1f]*>)?|</(?:{tags})[\s\x1c-\x1f]*>)"
    ))
}

/// Strip XML markup tags and collect `(type, value)` pairs in one pass. `value` is a wrapping
/// tag's inner text, else its first quoted attribute, else `""`. Square brackets are left alone.
#[crate::host(export(pid = false))]
pub fn extract_and_strip(text: &str, xml_tags: &[&str]) -> (String, Vec<(String, String)>) {
    // `xml_tags=[]` must be the identity.
    if xml_tags.is_empty() {
        return (text.to_string(), Vec::new());
    }
    let pattern = if xml_tags == ALL_MARKUP_TAGS.as_slice() {
        Cow::Borrowed(&*ALL_MARKUP_RE)
    } else {
        Cow::Owned(strip_pattern(xml_tags))
    };
    let mut tags = Vec::new();
    // Iterate to a fixed point so nested wrapping tags are fully removed (#6116); each pass
    // removes at least the delimiters. `tags` accumulates across passes.
    let mut clean = text.to_string();
    loop {
        let next = sub(&pattern, &clean, |m| {
            let Some(tag) = m.name("tag") else {
                return dedup_removal_space(m, &clean, ""); // lone closing tag
            };
            let inner = m.name("inner").map(|i| i.as_str());
            let value = match inner {
                Some(i) if !py_strip(i).is_empty() => py_strip(i).to_string(),
                _ => {
                    let attrs = m.name("attrs").map_or("", |a| a.as_str());
                    VALUE_ATTR_RE
                        .captures(attrs)
                        .expect("unlimited backtracking")
                        .map_or_else(String::new, |c| c[1].to_string())
                }
            };
            tags.push((tag.as_str().to_string(), value));
            // wrapping tags keep their inner content; self-closing/lone tags vanish
            dedup_removal_space(m, &clean, inner.unwrap_or(""))
        });
        if next == clean {
            return (clean, tags);
        }
        clean = next;
    }
}
