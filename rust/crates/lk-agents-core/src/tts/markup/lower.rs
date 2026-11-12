//! A provider's markup, from the lexer's tokens: `<expr>` markers lowered to the provider's own
//! syntax, its native tags in their final form, botched tags dropped. A wrapping marker wraps only
//! when its closer is in the same sentence; otherwise its delimiters go and its words stay.

use lk_agents_abi::tts::markup::MarkerKind;

use crate::lexer::{EXPR, Kind, Lexer, Rules, Tag, Token, Vocabulary};
use crate::tokenize::py_strip;
use crate::tts::markup::provider::{
    FISHAUDIO_DRAMA_SPANS, FISHAUDIO_TONES, GEMINI_PAUSE, GEMINI_SOUNDS, Provider, ProviderExt,
    XAI_WRAPPING, cartesia_prosody, fishaudio_sound_alias, gemini_sound_alias, xai_sound_alias,
};

fn rules(provider: Provider) -> Rules {
    Rules { tags: Some(Vocabulary::of(Some(provider))), ..Rules::default() }
}

/// A delta with its settled tags in canonical form, the point markers self-closed, so the sentence
/// splitter keeps only real wrappers open. A tag the delta ends inside stays as written: the next
/// delta completes it in the splitter's buffer.
pub(crate) fn normalize(provider: Option<Provider>, text: &str) -> String {
    let Some(provider) = provider else { return text.to_owned() };
    if !text.contains('<') {
        return text.to_owned();
    }
    let mut lexer = Lexer::new(rules(provider));
    let tokens = lexer.push(text);
    let held = text.len() - lexer.held();
    let mut out: String = tokens
        .iter()
        .map(|t| match &t.kind {
            Kind::Tag(tag) => closed(provider, tag).render(),
            _ => t.raw.clone(),
        })
        .collect();
    out.push_str(&text[held..]);
    out
}

/// `tag` with a point marker, or a native the provider writes self-closed, self-closed.
fn closed(provider: Provider, tag: &Tag) -> Tag {
    let point = if tag.name == EXPR {
        matches!(
            tag.attr("type").and_then(MarkerKind::from_wire),
            Some(MarkerKind::Expression | MarkerKind::Break | MarkerKind::Sound)
        )
    } else {
        provider.self_closing_tags().contains(&tag.name.as_str())
    };
    Tag { self_closing: tag.self_closing || (point && !tag.closing), ..tag.clone() }
}

/// One sentence in `provider`'s final syntax.
pub(crate) fn convert(provider: Option<Provider>, text: &str) -> String {
    let Some(provider) = provider else { return text.to_owned() };
    if !text.contains('<') {
        return text.to_owned();
    }
    let tokens = Lexer::all(rules(provider), text);
    Lowering { provider, out: Text::default(), frames: Vec::new() }.run(&tokens)
}

/// What a wrapper does with its words.
#[derive(Debug, Clone, Copy)]
enum Inner {
    Keep,
    Upper,
    Trim,
}

/// An open wrapper: what it emits around its words.
struct Frame {
    name: String,
    prefix: String,
    suffix: String,
    inner: Inner,
    words: Text,
}

/// Output where a removed marker between two blanks leaves one.
#[derive(Default)]
struct Text {
    s: String,
    removed: bool,
}

impl Text {
    fn push(&mut self, text: &str) {
        let text = if self.removed && self.s.ends_with([' ', '\t']) {
            text.trim_start_matches([' ', '\t'])
        } else {
            text
        };
        if !text.is_empty() {
            self.removed = false;
        }
        self.s.push_str(text);
    }

    fn remove(&mut self) {
        self.removed = true;
    }
}

struct Lowering {
    provider: Provider,
    out: Text,
    frames: Vec<Frame>,
}

impl Lowering {
    fn run(mut self, tokens: &[Token]) -> String {
        for (i, token) in tokens.iter().enumerate() {
            match &token.kind {
                Kind::Tag(tag) => self.tag(tag, &tokens[i + 1..]),
                Kind::Hidden(_) => self.text().remove(),
                _ => self.text().push(&token.raw),
            }
        }
        // a frame is only opened when its closer follows, so none is left
        self.out.s
    }

    fn text(&mut self) -> &mut Text {
        match self.frames.last_mut() {
            Some(frame) => &mut frame.words,
            None => &mut self.out,
        }
    }

    fn emit(&mut self, s: &str) {
        if s.is_empty() {
            self.text().remove();
        } else {
            self.text().push(s);
        }
    }

    fn tag(&mut self, tag: &Tag, rest: &[Token]) {
        if tag.closing {
            if self.frames.last().is_some_and(|f| f.name == tag.name) {
                let frame = self.frames.pop().expect("checked above");
                let words = match frame.inner {
                    Inner::Keep => frame.words.s,
                    Inner::Upper => frame.words.s.to_uppercase(),
                    Inner::Trim => py_strip(&frame.words.s).to_owned(),
                };
                self.emit(&format!("{}{words}{}", frame.prefix, frame.suffix));
            } else if tag.name != EXPR && as_written(self.provider, &tag.name) {
                self.emit(&tag.render());
            } else {
                // a stray closer is never spoken
                self.text().remove();
            }
            return;
        }
        if tag.name == EXPR {
            self.expr(tag, rest);
        } else {
            self.native(tag, rest);
        }
    }

    /// Whether a closer `rest` holds matches this opener, before another opener of the name.
    fn closes(name: &str, rest: &[Token]) -> bool {
        rest.iter()
            .filter_map(|t| match &t.kind {
                Kind::Tag(tag) if tag.name == name && !tag.self_closing => Some(tag.closing),
                _ => None,
            })
            .next()
            .unwrap_or(false)
    }

    fn open(&mut self, name: &str, prefix: String, suffix: String, inner: Inner) {
        self.frames.push(Frame {
            name: name.to_owned(),
            prefix,
            suffix,
            inner,
            words: Text::default(),
        });
    }

    fn expr(&mut self, tag: &Tag, rest: &[Token]) {
        let p = self.provider;
        let kind = tag.attr("type").and_then(MarkerKind::from_wire);
        let label = tag.attr("label").unwrap_or("");
        let wraps = matches!(kind, Some(MarkerKind::Prosody | MarkerKind::Spell));
        if wraps && !tag.self_closing {
            if !Self::closes(EXPR, rest) {
                // its closer is not in this sentence: the words without the marker
                self.text().remove();
                return;
            }
            let (prefix, suffix, inner) = wrapper(p, kind, label);
            self.open(EXPR, prefix, suffix, inner);
            return;
        }
        let lowered = match kind {
            // Gemini lifts an expression out of the text as the part's style
            Some(MarkerKind::Expression) if p == Provider::Gemini => closed(p, tag).render(),
            Some(MarkerKind::Expression) => expression(p, label),
            Some(MarkerKind::Sound) => sound(p, label),
            Some(MarkerKind::Break) => pause(p, label),
            Some(MarkerKind::Prosody) => point_prosody(p, label),
            Some(MarkerKind::Spell) | None => String::new(),
        };
        self.emit(&lowered);
    }

    fn native(&mut self, tag: &Tag, rest: &[Token]) {
        let p = self.provider;
        let value = tag.attr("value").unwrap_or("");
        let lowered = match (p, tag.name.as_str()) {
            (
                Provider::Inworld | Provider::Xai | Provider::FishAudioDrama,
                "expression" | "sound",
            ) => {
                format!("[{value}]")
            }
            (Provider::FishAudio, "expression") => fish_expression(value),
            (Provider::FishAudio, "sound") => format!("[{value}]"),
            (Provider::Xai | Provider::FishAudio | Provider::FishAudioDrama, "break") => {
                pause(p, tag.attr("time").unwrap_or(""))
            }
            (Provider::FishAudio, "emphasis") => {
                if Self::closes("emphasis", rest) {
                    self.open("emphasis", "[emphasis] ".into(), String::new(), Inner::Trim);
                } else {
                    self.text().remove();
                }
                return;
            }
            (Provider::FishAudio, _) => String::new(),
            _ => closed(p, tag).render(),
        };
        self.emit(&lowered);
    }
}

/// Whether `provider` takes its native tag `name` as written, rather than in a form of its own.
fn as_written(provider: Provider, name: &str) -> bool {
    match provider {
        Provider::Cartesia | Provider::Gemini => true,
        Provider::Inworld => name == "break",
        Provider::Xai | Provider::FishAudioDrama => {
            !matches!(name, "expression" | "sound" | "break")
        }
        // Fish's native dialect is brackets
        Provider::FishAudio => false,
    }
}

/// A paired prosody or spell marker in `provider`'s syntax: prefix, suffix and what its words get.
fn wrapper(provider: Provider, kind: Option<MarkerKind>, label: &str) -> (String, String, Inner) {
    let label = py_strip(label).to_lowercase();
    let none = (String::new(), String::new(), Inner::Keep);
    if kind == Some(MarkerKind::Spell) {
        return if provider == Provider::Cartesia {
            ("<spell>".into(), "</spell>".into(), Inner::Keep)
        } else {
            none
        };
    }
    match provider {
        Provider::Xai => {
            let native = label.replace(' ', "-");
            if XAI_WRAPPING.contains(&native.as_str()) {
                (format!("<{native}>"), format!("</{native}>"), Inner::Keep)
            } else {
                none
            }
        }
        // not advertised for Inworld; salvage a stray one as a delivery hint
        Provider::Inworld => (format!("[{label}]"), String::new(), Inner::Keep),
        Provider::Cartesia => {
            (cartesia_prosody(&label).unwrap_or("").into(), String::new(), Inner::Keep)
        }
        Provider::FishAudio if label == "emphasis" => {
            ("[emphasis] ".into(), String::new(), Inner::Trim)
        }
        // tone controls are prefix markers: `[whispering]` steers the words after it
        Provider::FishAudio if FISHAUDIO_TONES.contains(&label.as_str()) => {
            (format!("[{label}] "), String::new(), Inner::Keep)
        }
        Provider::FishAudio => none,
        Provider::FishAudioDrama => {
            let span = match label.as_str() {
                "whispering" => "whisper",
                "hurried" => "fast",
                label => label,
            };
            if FISHAUDIO_DRAMA_SPANS.contains(&span) {
                (format!("<{span}>"), format!("</{span}>"), Inner::Keep)
            } else {
                none
            }
        }
        // capitalizing is Gemini's one in-text stress control
        Provider::Gemini if label == "emphasis" => (String::new(), String::new(), Inner::Upper),
        Provider::Gemini => none,
    }
}

fn expression(provider: Provider, label: &str) -> String {
    match provider {
        // Cartesia's discrete emotion vocabulary (instructions list it)
        Provider::Cartesia => format!("<emotion value=\"{label}\"/>"),
        Provider::Inworld | Provider::FishAudioDrama => format!("[{label}]"),
        Provider::FishAudio => fish_expression(label),
        // xAI has no free-form delivery descriptions; Gemini is handled before
        Provider::Xai | Provider::Gemini => String::new(),
    }
}

/// Prefix "very" (steers Fish harder), never doubled when the LLM already wrote "very sad".
fn fish_expression(value: &str) -> String {
    let value = py_strip(value);
    if !value.is_empty() && !value.to_lowercase().starts_with("very ") {
        format!("[very {value}]")
    } else {
        format!("[{value}]")
    }
}

fn sound(provider: Provider, label: &str) -> String {
    match provider {
        Provider::Cartesia => String::new(),
        Provider::Inworld => format!("[{label}]"),
        Provider::Xai => format!("[{}]", xai_sound_alias(&label.to_lowercase()).unwrap_or(label)),
        Provider::FishAudio | Provider::FishAudioDrama => {
            format!("[{}]", fishaudio_sound_alias(&label.to_lowercase()).unwrap_or(label))
        }
        Provider::Gemini => {
            let l = py_strip(label).to_lowercase();
            let l = gemini_sound_alias(&l).unwrap_or(&l);
            if GEMINI_SOUNDS.contains(&l) { format!("<{l}>") } else { String::new() }
        }
    }
}

/// A pause of `time` (`"1s"`, `"300ms"`) in `provider`'s syntax.
fn pause(provider: Provider, time: &str) -> String {
    let long = break_seconds(time) >= 1.0;
    let pick = |short: &str, long_: &str| (if long { long_ } else { short }).to_owned();
    match provider {
        Provider::Gemini => format!("<{GEMINI_PAUSE}>"),
        Provider::Xai => pick("[pause]", "[long-pause]"),
        Provider::FishAudio => pick("[break]", "[long-break]"),
        Provider::FishAudioDrama => pick("[short pause]", "[pause]"),
        Provider::Cartesia | Provider::Inworld => format!("<break time=\"{time}\"/>"),
    }
}

/// A self-closed prosody marker: a point control for Cartesia, a tone prefix for Fish.
fn point_prosody(provider: Provider, label: &str) -> String {
    let label = py_strip(label).to_lowercase();
    match provider {
        Provider::Cartesia => cartesia_prosody(&label).unwrap_or("").to_owned(),
        Provider::FishAudio if FISHAUDIO_TONES.contains(&label.as_str()) => format!("[{label}]"),
        _ => String::new(),
    }
}

/// The duration in a break's time, `0.0` when it does not parse. `rstrip("s")` strips every
/// trailing `s`, as in the reference.
fn break_seconds(raw: &str) -> f64 {
    let raw = py_strip(raw).to_lowercase();
    let parsed = match raw.strip_suffix("ms") {
        Some(head) => py_float(head).map(|v| v / 1000.0),
        None => py_float(raw.trim_end_matches('s')),
    };
    parsed.unwrap_or(0.0)
}

/// CPython's `float(str)`: surrounding whitespace and `_` separators are allowed. Deliberate
/// divergence: non-ASCII Unicode digits fail to parse (CPython transliterates them).
fn py_float(s: &str) -> Option<f64> {
    let s = py_strip(s);
    if s.is_empty() {
        return None;
    }
    if s.contains('_') {
        let b = s.as_bytes();
        for (i, c) in b.iter().enumerate() {
            if *c == b'_'
                && !(i > 0
                    && i + 1 < b.len()
                    && b[i - 1].is_ascii_digit()
                    && b[i + 1].is_ascii_digit())
            {
                return None;
            }
        }
        return s.replace('_', "").parse().ok();
    }
    s.parse().ok()
}
