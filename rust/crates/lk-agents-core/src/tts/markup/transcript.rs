use lk_agents_abi::tts::markup::{EMOTION_TAG, ExpressiveTag, MarkerKind, TranscriptionExpression};
use lk_agents_abi::{ATTRIBUTE_TRANSCRIPTION_EXPRESSION, TranscriptChunk};

use crate::lexer::{EXPR, Kind, Lexer, Projector, Rules, Tag, Token, View, Vocabulary};
use crate::tts::markup::mood::{DEFAULT_MOOD, match_mood};
use crate::tts::markup::provider::ALL_MARKUP_TAGS;

/// Every provider's tags: a transcript sink that does not know the voice's provider.
fn every_provider() -> Vocabulary {
    Vocabulary::names(ALL_MARKUP_TAGS.iter().copied())
}

fn tag_rules(vocab: Vocabulary) -> Rules {
    Rules { tags: Some(vocab), ..Rules::default() }
}

const SHOWN: View = View::Display { keep_emoji: true };

/// The marker a tag published: an expr marker's type and label, a native tag's name and value.
fn expressive(tag: &Tag) -> Option<ExpressiveTag> {
    if tag.closing {
        return None;
    }
    Some(if tag.name == EXPR {
        ExpressiveTag::new(tag.attr("type").unwrap_or(""), tag.attr("label").unwrap_or(""))
    } else {
        let value = tag.attrs.first().map_or("", |(_, v)| v.as_str());
        ExpressiveTag::new(tag.name.clone(), value)
    })
}

/// The delivery a tag asks for from here on: an expression marker's label, or a native emotion or
/// expression tag's value. What the room publishes as `lk.expression`.
pub fn delivery(tag: &Tag) -> Option<String> {
    let marker = expressive(tag)?;
    let kind = MarkerKind::from_wire(&marker.kind);
    (kind == Some(MarkerKind::Expression) || marker.kind == EMOTION_TAG).then_some(marker.value)
}

fn tags_of(tokens: &[Token]) -> Vec<ExpressiveTag> {
    tokens
        .iter()
        .filter_map(|t| match &t.kind {
            Kind::Tag(tag) => expressive(tag),
            _ => None,
        })
        .collect()
}

/// `text` without the tags of `vocab`, and the markers they carried, in order.
fn split(vocab: Vocabulary, text: &str) -> (String, Vec<ExpressiveTag>) {
    if !text.contains('<') {
        return (text.to_owned(), Vec::new());
    }
    let tokens = Lexer::all(tag_rules(vocab), text);
    (Projector::new(SHOWN).text(&tokens), tags_of(&tokens))
}

/// Strip every provider's markup: the sink does not know the provider. Square brackets are prose
/// here and are left alone.
pub fn split_all_markup(text: &str) -> (String, Vec<ExpressiveTag>) {
    split(every_provider(), text)
}

/// [`split_all_markup`] returning only the clean text (tags discarded).
pub fn strip_all_markup(text: &str) -> String {
    split_all_markup(text).0
}

/// Strip only the `<expr/>` dialect, leaving all other markup untouched.
pub fn strip_expr_markup(text: &str) -> String {
    split(Vocabulary::LiveKit, text).0
}

/// Only the `<expr/>` dialect stripped, with its markers: the provider-native tags stay in the
/// words. With markers the text loses its trailing blanks.
pub fn split_expr_markup(text: &str) -> (String, Vec<ExpressiveTag>) {
    let (clean, tags) = split(Vocabulary::LiveKit, text);
    if tags.is_empty() {
        return (text.to_owned(), tags);
    }
    (clean.trim_end_matches([' ', '\t']).to_owned(), tags)
}

/// A segment's leading expression/emotion tag.
fn leading_expression(tags: &[ExpressiveTag]) -> Option<&ExpressiveTag> {
    tags.iter().find(|t| {
        MarkerKind::from_wire(&t.kind) == Some(MarkerKind::Expression) || t.kind == EMOTION_TAG
    })
}

/// A segment's leading expression as `<expr/>` markup, for a transcript whose words never carried
/// it (a voice's stamped words): the room reads it back as the same `lk.expression`.
pub fn expression_markup(tags: &[ExpressiveTag]) -> Option<String> {
    let expression = leading_expression(tags)?;
    let kind = MarkerKind::Expression.as_str();
    Some(format!("<expr type=\"{kind}\" label=\"{}\"/>", expression.value))
}

/// The `lk.expression` attribute from a segment's leading expression/emotion tag.
pub fn expression_attribute(tags: &[ExpressiveTag]) -> Option<(&'static str, String)> {
    delivery_attribute(&leading_expression(tags)?.value)
}

/// The `lk.expression` attribute for a delivery label.
pub fn delivery_attribute(label: &str) -> Option<(&'static str, String)> {
    // `match_mood`'s fallback is `DEFAULT_MOOD`, so `"mood"` is never null on the wire
    let mood = match_mood(label, Some(DEFAULT_MOOD)).unwrap_or(DEFAULT_MOOD);
    let value = TranscriptionExpression { expression: label, mood: mood.as_str() };
    let json = serde_json::to_string(&value).ok()?;
    Some((ATTRIBUTE_TRANSCRIPTION_EXPRESSION, json))
}

/// [`expression_attribute`] as the one-entry `{ATTRIBUTE_TRANSCRIPTION_EXPRESSION: json}` dict the
/// reference returns.
pub fn expression_attribute_map(
    tags: Vec<ExpressiveTag>,
) -> Option<std::collections::BTreeMap<String, String>> {
    attribute_map(expression_attribute(&tags))
}

fn attribute_map(
    pair: Option<(&'static str, String)>,
) -> Option<std::collections::BTreeMap<String, String>> {
    pair.map(|(key, json)| std::collections::BTreeMap::from([(key.to_owned(), json)]))
}

/// One transcript segment's markup stripper: a tag split across chunks is held until it closes, a
/// cut one never shown. A trailing blank is held until more text or the segment's end, so text
/// cut between two chunks never ends in a separator.
#[derive(Debug)]
pub struct TranscriptMarkupStripper {
    lexer: Lexer,
    shown: Projector,
    tags: Vec<ExpressiveTag>,
    blank: String,
}

impl Default for TranscriptMarkupStripper {
    fn default() -> Self {
        TranscriptMarkupStripper {
            lexer: Lexer::new(tag_rules(every_provider())),
            shown: Projector::new(SHOWN),
            tags: Vec::new(),
            blank: String::new(),
        }
    }
}

impl TranscriptMarkupStripper {
    pub fn new() -> Self {
        Self::default()
    }

    fn take(&mut self, tokens: Vec<Token>) -> String {
        self.tags.extend(tags_of(&tokens));
        let text = self.shown.text(&tokens);
        if text.is_empty() {
            return text;
        }
        let mut out = std::mem::take(&mut self.blank);
        out.push_str(&text);
        let kept = out.trim_end_matches([' ', '\t']).len();
        self.blank = out.split_off(kept);
        out
    }

    /// Feed a chunk; return the clean text ready to emit (may be empty).
    pub fn push(&mut self, text: &str) -> String {
        let tokens = self.lexer.push(text);
        self.take(tokens)
    }

    /// The segment ended: whatever was held settles.
    pub fn flush(&mut self) -> String {
        let tokens = self.lexer.finish();
        let mut out = self.take(tokens);
        out.push_str(&std::mem::take(&mut self.blank));
        out
    }

    /// The markup tags stripped so far, in document order.
    pub fn tags(&self) -> &[ExpressiveTag] {
        &self.tags
    }

    /// The `lk.expression` attribute for the tags stripped so far, if any.
    pub fn expression_attribute(&self) -> Option<(&'static str, String)> {
        expression_attribute(&self.tags)
    }

    /// [`Self::expression_attribute`] as the host's one-entry dict.
    pub fn expression_attribute_map(&self) -> Option<std::collections::BTreeMap<String, String>> {
        attribute_map(self.expression_attribute())
    }
}

const MAX_HELD_CHARS: usize = 256;

/// `held` carries an unclosed span across calls; call once more with `end_of_stream = true`.
pub fn drop_bracket_cues(
    tokens: Vec<TranscriptChunk>,
    held: &mut Vec<TranscriptChunk>,
    end_of_stream: bool,
) -> Vec<TranscriptChunk> {
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

    let mut out: Vec<TranscriptChunk> = Vec::new();
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

/// [`drop_bracket_cues`] as a plan over token texts, for a host whose `TranscriptChunk` cannot cross.
/// Returns `(emit, keep)`; `None` text means reuse the host's token object.
pub fn drop_bracket_cues_plan(texts: &[String], end_of_stream: bool) -> (CuePlan, CuePlan) {
    let tokens: Vec<TranscriptChunk> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| TranscriptChunk {
            text: text.clone(),
            start_time: Some(i as f64),
            ..Default::default()
        })
        .collect();
    let mut held: Vec<TranscriptChunk> = Vec::new();
    let out = drop_bracket_cues(tokens, &mut held, end_of_stream);

    let plan = |tokens: Vec<TranscriptChunk>| -> CuePlan {
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

fn retext(token: &TranscriptChunk, text: String) -> TranscriptChunk {
    TranscriptChunk { text, ..token.clone() }
}

fn retext_owned(token: TranscriptChunk, text: String) -> TranscriptChunk {
    TranscriptChunk { text, ..token }
}
