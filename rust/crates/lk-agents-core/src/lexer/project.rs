//! Views: what each consumer reads of the tokens.

use std::borrow::Cow;

use super::{Kind, Md, Token};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// The model's text with its reasoning hidden: what the LLM stage hands on.
    Model,
    /// What the voice says: no markup (the TTS lowers it on its own), no markdown syntax, no
    /// emoji, no struck text, no fenced code.
    Spoken,
    /// What the TTS is sent: the spoken view with each recognized tag in its canonical spelling,
    /// for the provider to lower.
    Synthesis,
    /// The transcript: markdown and emoji as written for the frontend to render, markup gone.
    Display,
    /// The history: plain words, no markdown, markup or struck text; code, emoji and a link's url
    /// stay.
    Plain,
}

impl Token {
    pub fn view(&self, view: View) -> Cow<'_, str> {
        let raw = Cow::Borrowed(self.raw.as_str());
        let none = Cow::Borrowed("");
        let said = matches!(view, View::Spoken | View::Synthesis);
        // the synthesis view is the spoken one, with each tag in its canonical spelling
        match (&self.kind, view) {
            (Kind::Hidden(_), _) => none,
            (_, View::Model) => raw,
            (Kind::Tag(tag), View::Synthesis) => Cow::Owned(tag.render()),
            (Kind::Tag(_), _) => none,
            (Kind::Prose, _) if said && (self.ctx.struck || self.ctx.fenced) => none,
            (Kind::Prose, View::Plain) if self.ctx.struck => none,
            (Kind::Prose, _) => raw,
            (Kind::Emoji | Kind::Markdown(_), _) if said => none,
            (Kind::Emoji, _) => raw,
            (Kind::Markdown(_), View::Display) => raw,
            (Kind::Markdown(Md::LinkClose { url }), View::Plain) => Cow::Owned(format!(" ({url})")),
            (Kind::Markdown(_), _) => none,
        }
    }
}

/// One view over a stream of tokens. Where something was removed between two blanks, one blank is
/// enough; at the start, none.
#[derive(Debug, Clone)]
pub struct Projector {
    view: View,
    removed: bool,
    last: Option<char>,
}

impl Projector {
    pub fn new(view: View) -> Projector {
        Projector { view, removed: false, last: None }
    }

    pub fn push(&mut self, token: &Token) -> String {
        let text = token.view(self.view);
        if text.is_empty() {
            self.removed = true;
            return String::new();
        }
        let text = if self.removed && self.last.is_none_or(|c| c == ' ' || c == '\t') {
            text.trim_start_matches([' ', '\t'])
        } else {
            &text
        };
        if text.is_empty() {
            return String::new();
        }
        self.removed = false;
        self.last = text.chars().next_back();
        text.to_owned()
    }

    pub fn text(&mut self, tokens: &[Token]) -> String {
        tokens.iter().map(|t| self.push(t)).collect()
    }
}
