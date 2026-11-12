//! What a model writes that is not for the person: its reasoning, tool calls it wrote as text, its
//! chat template's turn structure (role headers, turns of other roles it went on to invent, gpt-oss's
//! non-final channels) and control tokens. These are the model's special tokens, matched literally
//! from a per-model [`ModelSyntax`].

use super::{Hidden, Kind, Out, Token};

/// A span hidden with its delimiters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub open: &'static str,
    /// `None`: the span runs to the end of the reply.
    pub close: Option<&'static str>,
    pub why: Hidden,
}

/// The opener of one role's turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Turn {
    pub open: &'static str,
    /// The assistant's turn is said; any other role's is the model inventing the next message.
    pub said: bool,
}

/// A chat template's turn structure: `open`, a header up to `header_end`, the content, an end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Turns {
    /// Longest first, so `<|im_start|>assistant` wins over a bare `<|im_start|>`.
    pub opens: &'static [Turn],
    /// The header after an opener (a channel, a recipient), never said; `None` has none.
    pub header_end: Option<&'static str>,
    pub ends: &'static [&'static str],
    /// gpt-oss: a header naming a channel is said only for `final`.
    pub channel: Option<&'static str>,
}

/// One model's special syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSyntax {
    pub spans: &'static [Span],
    /// The chat template already opened the first span: the reply starts inside it and only its
    /// close arrives.
    pub starts_open: bool,
    pub turns: Option<Turns>,
    /// Other template tokens, dropped wherever they appear.
    pub controls: &'static [&'static str],
}

impl ModelSyntax {
    pub const THINK: Span =
        Span { open: "<think>", close: Some("</think>"), why: Hidden::Reasoning };

    /// `<think>...</think>`, what a model with no entry of its own gets.
    pub const DEFAULT: ModelSyntax = ModelSyntax {
        spans: &[ModelSyntax::THINK],
        starts_open: false,
        turns: None,
        controls: &[],
    };

    /// gpt-oss's harmony format.
    pub const HARMONY: ModelSyntax = ModelSyntax {
        spans: &[],
        starts_open: false,
        turns: Some(Turns {
            opens: &[
                Turn { open: "<|start|>assistant", said: true },
                Turn { open: "<|start|>", said: false },
            ],
            header_end: Some("<|message|>"),
            ends: &["<|end|>", "<|return|>", "<|call|>"],
            channel: Some("<|channel|>"),
        }),
        controls: &["<|channel|>", "<|constrain|>", "<|message|>"],
    };

    /// ChatML (Qwen, Hermes and others).
    pub const CHATML: Turns = Turns {
        opens: &[
            Turn { open: "<|im_start|>assistant", said: true },
            Turn { open: "<|im_start|>", said: false },
        ],
        header_end: Some("\n"),
        ends: &["<|im_end|>", "<|endoftext|>"],
        channel: None,
    };

    /// Llama 3.
    pub const LLAMA3: Turns = Turns {
        opens: &[
            Turn { open: "<|start_header_id|>assistant<|end_header_id|>", said: true },
            Turn { open: "<|start_header_id|>", said: false },
        ],
        header_end: None,
        ends: &["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"],
        channel: None,
    };

    /// Every literal this syntax reacts to.
    fn delimiters(&self) -> impl Iterator<Item = &'static str> + '_ {
        let turns = self.turns.iter().flat_map(|t| {
            t.opens.iter().map(|o| o.open).chain(t.ends.iter().copied()).chain(t.channel)
        });
        self.spans
            .iter()
            .flat_map(|s| std::iter::once(s.open).chain(s.close))
            .chain(turns)
            .chain(self.controls.iter().copied())
    }

    /// The characters a delimiter may start with: prose stops there.
    pub(super) fn starts(&self) -> Vec<char> {
        self.delimiters().filter_map(|d| d.chars().next()).collect()
    }
}

/// Where the reply is in its template's turns.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) enum Place {
    /// No turn structure seen: the reply as the API hands it, said.
    #[default]
    Plain,
    /// A header, collected to decide its channel.
    Header { said: bool, header: String },
    /// Content; `fresh` until its first character, so the blank lines after a header go.
    Content { said: bool, fresh: bool },
    /// After an end token, until another turn opens.
    Closed,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct ModelState {
    /// The span we are inside.
    span: Option<Span>,
    place: Place,
}

impl ModelState {
    pub(super) fn new(syntax: &ModelSyntax) -> ModelState {
        ModelState {
            span: syntax.spans.first().copied().filter(|_| syntax.starts_open),
            place: Place::Plain,
        }
    }
}

/// The model's answer at a position.
pub(super) enum Step {
    /// `len` bytes were the model's own syntax, hidden.
    Hid(usize),
    /// Not the model's syntax: read on.
    Pass,
    /// Could still become a delimiter: hold.
    Hold,
}

fn hide(out: &mut Out, raw: &str, why: Hidden) {
    out.push(Token { kind: Kind::Hidden(why), raw: raw.to_owned(), ctx: Default::default() });
}

/// The longest proper prefix of `delimiter` that `text` ends with, in bytes.
pub(super) fn held_prefix_len(text: &str, delimiter: &str) -> usize {
    (1..delimiter.len())
        .rev()
        .filter(|n| delimiter.is_char_boundary(*n))
        .find(|n| text.ends_with(&delimiter[..*n]))
        .unwrap_or(0)
}

/// Hide `s` up to the first of `stops`, holding a partial one at the tail.
fn hide_until(out: &mut Out, s: &str, stops: &[&str], at_end: bool, why: Hidden) -> Step {
    let next = stops.iter().filter_map(|d| s.find(d)).min();
    let mut len = next.unwrap_or(s.len());
    if next.is_none() && !at_end {
        len -= stops.iter().map(|d| held_prefix_len(s, d)).max().unwrap_or(0);
    }
    if len == 0 {
        return Step::Hold;
    }
    hide(out, &s[..len], why);
    Step::Hid(len)
}

/// Read the text at the start of `s`.
pub(super) fn step(
    syntax: &ModelSyntax,
    state: &mut ModelState,
    s: &str,
    at_end: bool,
    out: &mut Out,
) -> Step {
    if let Some(span) = state.span {
        let Some(close) = span.close else {
            hide(out, s, span.why);
            return Step::Hid(s.len());
        };
        return match s.find(close) {
            Some(r) => {
                hide(out, &s[..r + close.len()], span.why);
                state.span = None;
                Step::Hid(r + close.len())
            }
            None => hide_until(out, s, &[close], at_end, span.why),
        };
    }
    // a longer delimiter may still arrive: `<|im_start|>` could become `<|im_start|>assistant`
    if !at_end && syntax.delimiters().any(|d| d.len() > s.len() && d.starts_with(s)) {
        return Step::Hold;
    }
    if let Some(turns) = &syntax.turns
        && let Some(step) = turn_step(turns, state, s, at_end, out)
    {
        return step;
    }
    if let Some(span) = syntax.spans.iter().find(|sp| s.starts_with(sp.open)) {
        hide(out, span.open, span.why);
        state.span = Some(*span);
        return Step::Hid(span.open.len());
    }
    let strays =
        syntax.spans.iter().filter_map(|sp| sp.close).chain(syntax.controls.iter().copied());
    if let Some(d) = strays.into_iter().find(|d| s.starts_with(d)) {
        hide(out, d, Hidden::Control);
        return Step::Hid(d.len());
    }
    // part of one, at the end of the stream (held above until then; a whole one matched above)
    if syntax.delimiters().any(|d| d.starts_with(s)) {
        hide(out, s, Hidden::Control);
        return Step::Hid(s.len());
    }
    if let Place::Content { fresh: fresh @ true, .. } = &mut state.place {
        // the blank lines between a header and its content, hidden as they arrive
        let blank = s.len() - s.trim_start_matches(['\n', '\r', ' ']).len();
        if blank > 0 {
            hide(out, &s[..blank], Hidden::Control);
            return Step::Hid(blank);
        }
        *fresh = false;
    }
    Step::Pass
}

/// The turn structure's reading at `s`; `None` leaves it to the rest of the syntax.
fn turn_step(
    turns: &Turns,
    state: &mut ModelState,
    s: &str,
    at_end: bool,
    out: &mut Out,
) -> Option<Step> {
    // inside a header only its end acts
    if let Place::Header { said, header } = &mut state.place {
        let header_end = turns.header_end.expect("a header has an end");
        if let Some(r) = s.find(header_end) {
            header.push_str(&s[..r]);
            let said = *said && channel_is_final(turns, header);
            hide(out, &s[..r + header_end.len()], Hidden::Channel);
            state.place = Place::Content { said, fresh: true };
            return Some(Step::Hid(r + header_end.len()));
        }
        let step = hide_until(out, s, &[header_end], at_end, Hidden::Channel);
        if let Step::Hid(len) = step {
            header.push_str(&s[..len]);
        }
        return Some(step);
    }
    if let Some(turn) = turns.opens.iter().find(|t| s.starts_with(t.open)) {
        hide(out, turn.open, Hidden::Channel);
        state.place = match turns.header_end {
            Some(_) => Place::Header { said: turn.said, header: String::new() },
            None => Place::Content { said: turn.said, fresh: true },
        };
        return Some(Step::Hid(turn.open.len()));
    }
    // a channel with no turn opener: the server stripped `<|start|>assistant`
    if let Some(marker) = turns.channel.filter(|m| s.starts_with(m)) {
        hide(out, marker, Hidden::Channel);
        state.place = Place::Header { said: true, header: marker.to_owned() };
        return Some(Step::Hid(marker.len()));
    }
    if let Some(end) = turns.ends.iter().find(|e| s.starts_with(**e)) {
        hide(out, end, Hidden::Control);
        state.place = Place::Closed;
        return Some(Step::Hid(end.len()));
    }
    let opens: Vec<&str> = turns.opens.iter().map(|t| t.open).collect();
    match &mut state.place {
        Place::Plain | Place::Content { said: true, .. } | Place::Header { .. } => None,
        Place::Content { said: false, .. } => {
            let stops: Vec<&str> =
                opens.iter().copied().chain(turns.ends.iter().copied()).collect();
            Some(hide_until(out, s, &stops, at_end, Hidden::Channel))
        }
        Place::Closed => Some(hide_until(out, s, &opens, at_end, Hidden::Channel)),
    }
}

/// A header naming a channel is said only for `final`.
fn channel_is_final(turns: &Turns, header: &str) -> bool {
    let Some(marker) = turns.channel else { return true };
    match header.split_once(marker) {
        Some((_, rest)) => {
            rest.split(|c: char| c.is_whitespace() || c == '<').next() == Some("final")
        }
        None => true,
    }
}
