//! The provider-agnostic transcript side: strip every dialect, surface the leading expression, hold
//! a tag that straddles a chunk boundary.

use lk_agents_abi::{ATTRIBUTE_TRANSCRIPTION_EXPRESSION, TimedString};

use crate::tts::markup::expr::{ExpressiveTag, split_expr};
use crate::tts::markup::mood::{DEFAULT_MOOD, match_mood};
use crate::tts::markup::provider::ALL_MARKUP_TAGS;
use crate::tts::markup::strip::extract_and_strip;

/// Strip the union of every provider's expressive XML markup (provider-agnostic): the sinks no
/// longer know the provider. Square brackets are prose here and are left alone.
#[crate::host(export(pid = false))]
pub fn split_all_markup(text: &str) -> (String, Vec<ExpressiveTag>) {
    // Every markup shape starts with `<`: skip the scan for the common, markup-free chunk.
    if !text.contains('<') {
        return (text.to_string(), Vec::new());
    }
    let (text, expr_tags) = split_expr(text);
    let (clean, raw_tags) = extract_and_strip(&text, &ALL_MARKUP_TAGS);
    let mut tags = expr_tags;
    tags.extend(raw_tags.into_iter().map(|(t, v)| ExpressiveTag::new(t, v)));
    (clean, tags)
}

/// [`split_all_markup`] returning only the clean text (tags discarded).
#[crate::host(export(pid = false))]
pub fn strip_all_markup(text: &str) -> String {
    split_all_markup(text).0
}

/// Strip only the `<expr/>` dialect, leaving all other markup untouched.
#[crate::host(export(pid = false))]
pub fn strip_expr_markup(text: &str) -> String {
    split_expr(text).0
}

/// `{"expression", "mood"}` from a segment's leading expression/emotion tag.
pub fn expression_attribute(tags: &[ExpressiveTag]) -> Option<(&'static str, String)> {
    let expression = tags.iter().find(|t| t.kind == "expression" || t.kind == "emotion")?;
    // `match_mood`'s fallback is `DEFAULT_MOOD`, so `"mood"` is never null on the wire
    let mood = match_mood(&expression.value, Some(DEFAULT_MOOD)).unwrap_or(DEFAULT_MOOD);
    let mut json = String::from("{\"expression\":");
    push_json_string(&mut json, &expression.value);
    json.push_str(",\"mood\":");
    push_json_string(&mut json, mood.as_str());
    json.push('}');
    Some((ATTRIBUTE_TRANSCRIPTION_EXPRESSION, json))
}

/// [`expression_attribute`] as the one-entry `{ATTRIBUTE_TRANSCRIPTION_EXPRESSION: json}` dict the
/// reference returns.
#[crate::host(export(pid = false, name = "expression_attribute"))]
pub fn expression_attribute_map(tags: Vec<ExpressiveTag>) -> Option<std::collections::HashMap<String, String>> {
    attribute_map(expression_attribute(&tags))
}

fn attribute_map(pair: Option<(&'static str, String)>) -> Option<std::collections::HashMap<String, String>> {
    pair.map(|(key, json)| std::collections::HashMap::from([(key.to_owned(), json)]))
}

/// `ensure_ascii=True`: non-ASCII is escaped as `\uXXXX`, which a JSON writer would not do.
fn push_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c if (c as u32) < 0x10000 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => {
                let n = c as u32 - 0x10000;
                out.push_str(&format!("\\u{:04x}", 0xd800 + (n >> 10)));
                out.push_str(&format!("\\u{:04x}", 0xdc00 + (n & 0x3ff)));
            }
        }
    }
    out.push('"');
}

/// Stateful, provider-agnostic markup stripper for one transcript segment. A partial trailing tag
/// is held back until it closes.
#[crate::class(mutable)]
#[derive(Debug, Default)]
pub struct TranscriptMarkupStripper {
    buf: String,
    tags: Vec<ExpressiveTag>,
    seam_after_strip: bool,
}

/// Stateful, provider-agnostic markup stripper for one transcript segment.
#[crate::host(object(
    pid = false
))]
impl TranscriptMarkupStripper {
    #[host(new)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Strip `text` and record its tags. Trailing whitespace is held so a tag opening the next
    /// chunk is still deduped against it; `end_of_segment` releases it.
    fn consume(&mut self, text: &str, end_of_segment: bool) -> String {
        let owned;
        let text = if self.seam_after_strip && text.starts_with([' ', '\t']) {
            // A tag was stripped at the held whitespace: collapse it with the following run.
            owned = format!("{}{}", &text[..1], text[1..].trim_start_matches([' ', '\t']));
            owned.as_str()
        } else {
            text
        };

        let (clean, tags) = split_all_markup(text);
        let any_tags = !tags.is_empty();
        self.tags.extend(tags);

        let held = if end_of_segment { "" } else { &clean[rstrip_spaces_tabs(&clean).len()..] };
        // Only when this chunk ended on a tag (#6717). `endswith(">")` is the reference's
        // approximation, wrong for a trailing wrapping tag; kept for parity.
        self.seam_after_strip =
            any_tags && !held.is_empty() && rstrip_spaces_tabs(text).ends_with('>');

        let emit = clean[..clean.len() - held.len()].to_string();
        self.buf = held.to_string();
        emit
    }

    /// Is the buffer's tail a partial XML tag? Only a tag-shaped `<` is held (`"3 < 5"` must not
    /// stall, #6116), and never a `[` (#6590).
    fn has_open_tag(&self) -> bool {
        let Some(last_lt) = self.buf.rfind('<') else { return false };
        if self.buf.rfind('>').is_none_or(|gt| last_lt > gt) {
            return match self.buf[last_lt + 1..].chars().next() {
                None => true,
                Some('/') => true,
                Some(c) => c.is_alphabetic(),
            };
        }
        false
    }

    /// Feed a chunk; return the clean text ready to emit (may be empty).
    #[host(method)]
    pub fn push(&mut self, text: &str) -> String {
        self.buf.push_str(text);
        if self.has_open_tag() {
            return String::new();
        }
        let buf = std::mem::take(&mut self.buf);
        self.consume(&buf, false)
    }

    /// Drain any buffered text at segment end; return the remaining clean text.
    #[host(method)]
    pub fn flush(&mut self) -> String {
        if self.buf.is_empty() {
            return String::new();
        }
        let buf = std::mem::take(&mut self.buf);
        self.consume(&buf, true)
    }

    /// The markup tags stripped so far, in document order.
    #[host(getter)]
    pub fn tags(&self) -> &[ExpressiveTag] {
        &self.tags
    }

    /// The `lk.expression` attribute for the tags stripped so far, if any.
    pub fn expression_attribute(&self) -> Option<(&'static str, String)> {
        expression_attribute(&self.tags)
    }

    /// [`Self::expression_attribute`] as the host's one-entry dict.
    #[host(method(name = "expression_attribute"))]
    pub fn expression_attribute_map(&self) -> Option<std::collections::HashMap<String, String>> {
        attribute_map(self.expression_attribute())
    }
}

/// The reference's value, unexplained.
const MAX_HELD_CHARS: usize = 256;

/// `held` carries an unclosed span across calls; call once more with `end_of_stream = true`.
pub fn drop_bracket_cues(
    tokens: Vec<TimedString>,
    held: &mut Vec<TimedString>,
    end_of_stream: bool,
) -> Vec<TimedString> {
    // `tokens = held + tokens; held.clear()`
    let mut prefixed = std::mem::take(held);
    prefixed.extend(tokens);
    let tokens = prefixed;
    let text: Vec<char> = tokens.iter().flat_map(|t| t.text.chars()).collect();
    if !text.contains(&'[') {
        return tokens;
    }

    // `\[[^\]]*\]`, non-overlapping, over *character* indices
    let mut dropped = vec![false; text.len()];
    let mut i = 0usize;
    while i < text.len() {
        if text[i] != '[' {
            i += 1;
            continue;
        }
        let Some(rel) = text[i + 1..].iter().position(|c| *c == ']') else { break };
        let (mut start, mut end) = (i, i + 1 + rel + 1);
        i = end;
        // take one of the spaces the cue sat between, so it leaves a single separator
        if start > 0 && text[start - 1] == ' ' && (end == text.len() || text[end] == ' ') {
            start -= 1;
        } else if start == 0 && end < text.len() && text[end] == ' ' {
            end += 1;
        }
        dropped[start..end].fill(true);
    }

    // Hold from an unclosed `[`; past MAX_HELD_CHARS give up so a lone bracket cannot stall.
    let mut hold_from = text.len();
    if !end_of_stream {
        let open_at = text.iter().rposition(|c| *c == '[');
        let close_at = text.iter().rposition(|c| *c == ']');
        if let Some(open_at) = open_at
            && close_at.is_none_or(|c| open_at > c)
            && text.len() - open_at <= MAX_HELD_CHARS
        {
            hold_from = open_at;
        }
    }

    let mut out: Vec<TimedString> = Vec::new();
    let mut pos = 0usize;
    for token in tokens {
        let chars: Vec<char> = token.text.chars().collect();
        let emit: String = chars
            .iter()
            .enumerate()
            .filter(|(i, _)| pos + i < hold_from && !dropped[pos + i])
            .map(|(_, c)| *c)
            .collect();
        let keep: String = chars
            .iter()
            .enumerate()
            .filter(|(i, _)| pos + i >= hold_from)
            .map(|(_, c)| *c)
            .collect();
        pos += chars.len();
        if !emit.is_empty() {
            out.push(if emit == token.text { token.clone() } else { retext(&token, emit) });
        }
        if !keep.is_empty() {
            held.push(if keep == token.text { token } else { retext_owned(token, keep) });
        }
    }
    out
}

/// `None` when the token survives byte-for-byte.
pub type CuePlan = Vec<(usize, Option<String>)>;

/// [`drop_bracket_cues`] as a plan over token texts, for a host whose `TimedString` cannot cross.
/// Returns `(emit, keep)`; `None` text means reuse the host's token object.
#[crate::host(export(name = "drop_bracket_cues", pid = false))]
pub fn drop_bracket_cues_plan(texts: &[String], end_of_stream: bool) -> (CuePlan, CuePlan) {
    let tokens: Vec<TimedString> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| TimedString {
            text: text.clone(),
            start_time: Some(i as f64),
            ..Default::default()
        })
        .collect();
    let mut held: Vec<TimedString> = Vec::new();
    let out = drop_bracket_cues(tokens, &mut held, end_of_stream);

    let plan = |tokens: Vec<TimedString>| -> CuePlan {
        tokens
            .into_iter()
            .map(|token| {
                let index = token.start_time.unwrap_or_default() as usize;
                let unchanged = texts.get(index).is_some_and(|orig| *orig == token.text);
                (index, (!unchanged).then_some(token.text))
            })
            .collect()
    };
    (plan(out), plan(held))
}

fn retext(token: &TimedString, text: String) -> TimedString {
    TimedString { text, ..token.clone() }
}

fn retext_owned(token: TimedString, text: String) -> TimedString {
    TimedString { text, ..token }
}

/// The only two characters `TranscriptMarkupStripper` holds back.
fn rstrip_spaces_tabs(s: &str) -> &str {
    s.trim_end_matches([' ', '\t'])
}
