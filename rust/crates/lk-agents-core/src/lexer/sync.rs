//! The synchronizer's units: the transcript in source order, each piece either text the voice
//! says or text with no speech.
//!
//! Spoken text is paced by the voice (the pacer splits it into words); a silent unit (markdown
//! syntax, an emoji the voice skips, fenced code, struck text, a markup tag's delivery) costs
//! nothing and is shown as soon as everything before it has been heard. No unit is tied to
//! another, and nothing is held: a unit leaves with the token it came from.
//!
//! Joined, the units' `shown` is exactly the transcript view, and the words of the spoken units are
//! exactly the spoken view's words, however the text was chunked.

use super::{Kind, Projector, Tag, Token, View};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unit {
    /// Said, and shown as said.
    Said(String),
    /// Shown, or a markup tag's delivery taking effect, with no speech.
    Silent { shown: String, tags: Vec<Tag> },
}

impl Unit {
    pub fn shown(&self) -> &str {
        match self {
            Unit::Said(text) => text,
            Unit::Silent { shown, .. } => shown,
        }
    }
}

/// Units over one stream of tokens.
#[derive(Debug, Clone)]
pub struct Sync {
    said: Projector,
    shown: Projector,
}

impl Sync {
    pub fn new(said: View, shown: View) -> Sync {
        Sync { said: Projector::new(said), shown: Projector::new(shown) }
    }

    pub fn push(&mut self, tokens: &[Token]) -> Vec<Unit> {
        let mut out: Vec<Unit> = Vec::new();
        for token in tokens {
            // both views see every token, so each keeps its own blank collapsing
            let said = self.said.push(token);
            let shown = self.shown.push(token);
            let unit = match &token.kind {
                Kind::Tag(tag) => Unit::Silent { shown, tags: vec![tag.clone()] },
                _ if !said.is_empty() && !shown.is_empty() => Unit::Said(shown),
                _ if !shown.is_empty() => Unit::Silent { shown, tags: Vec::new() },
                _ => continue,
            };
            match (out.last_mut(), unit) {
                (Some(Unit::Said(last)), Unit::Said(more)) => last.push_str(&more),
                (Some(Unit::Silent { shown: last, .. }), Unit::Silent { shown: more, tags })
                    if tags.is_empty() =>
                {
                    last.push_str(&more);
                }
                (_, unit) => out.push(unit),
            }
        }
        out
    }
}
