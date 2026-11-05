//! The `<expr/>` dialect: split out of a transcript, lowered to each provider's native syntax. The
//! patterns are `tts/_provider_format.py`'s, transcribed as `strip.rs` describes.

use crate::tts::markup::provider::{
    FISHAUDIO_TONES, GEMINI_PAUSE, GEMINI_SOUNDS, Provider, ProviderExt, XAI_WRAPPING,
    cartesia_prosody, fishaudio_sound_alias, gemini_sound_alias, xai_sound_alias,
};
use crate::tts::markup::strip::{
    Captures, EXPRESSION_RE, LEADING_WS, WORD_BOUNDARY as B, compile, convert_expression_tags,
    dedup_removal_space, py_strip, sub,
};
use fancy_regex::Regex;
use std::sync::LazyLock;

pub use lk_agents_abi::tts::markup::ExpressiveTag;

// Every marker pattern captures the space before it as `pre`, so `dedup_removal_space` can drop it
// when the marker vanishes from between two spaces.
static EXPR_ATTR_RE: LazyLock<Regex> =
    LazyLock::new(|| compile(r#"([\p{L}\p{N}_-]+)[\s\x1c-\x1f]*=[\s\x1c-\x1f]*"([^"]*)""#));
/// Any `<expr ...>` or `<expr .../>` tag, open or self-closing.
static EXPR_OPEN_RE: LazyLock<Regex> =
    LazyLock::new(|| compile(&format!(r"{LEADING_WS}<expr{B}(?P<attrs>[^>]*?)/?[\s\x1c-\x1f]*>")));
static EXPR_CLOSE_RE: LazyLock<Regex> =
    LazyLock::new(|| compile(&format!(r"{LEADING_WS}</expr[\s\x1c-\x1f]*>")));
/// Self-closing markers only (the trailing `/` is required).
static EXPR_SELF_RE: LazyLock<Regex> =
    LazyLock::new(|| compile(&format!(r"{LEADING_WS}<expr{B}(?P<attrs>[^>]*?)/[\s\x1c-\x1f]*>")));
/// A wrapping marker (prosody/spell) and its span; non-greedy, instructed not to nest.
static EXPR_WRAP_RE: LazyLock<Regex> = LazyLock::new(|| {
    compile(&format!(
        r#"(?s){LEADING_WS}<expr{B}(?=[^>]*type="(?:prosody|spell)")(?P<attrs>[^>]*?)>(?P<inner>.*?)</expr[\s\x1c-\x1f]*>"#
    ))
});
/// A non-wrapping type the LLM forgot to self-close ([`normalize_markup`] fixes these).
static EXPR_UNCLOSED_RE: LazyLock<Regex> = LazyLock::new(|| {
    compile(&format!(
        r#"(<expr{B}(?=[^>]*type="(?:expression|break|sound)")[^>]*[^/>\s\x1c-\x1f])[\s\x1c-\x1f]*>"#
    ))
});
/// `_XAI_BREAK_RE` and `_FISHAUDIO_BREAK_RE`, which are the same pattern.
static BREAK_RE: LazyLock<Regex> =
    LazyLock::new(|| compile(r#"<break[\s\x1c-\x1f]+time="([^"]*)"[\s\x1c-\x1f]*/?>"#));
static FISHAUDIO_EMPHASIS_RE: LazyLock<Regex> = LazyLock::new(|| {
    compile(r"(?i)<emphasis(?:[\s\x1c-\x1f][^>]*)?>([^<]*)</emphasis[\s\x1c-\x1f]*>")
});

/// `pattern.sub(lambda m: _dedup_removal_space(m, repl(m)), text)`.
fn sub_expr(re: &Regex, text: &str, mut repl: impl FnMut(&Captures) -> String) -> String {
    sub(re, text, |m| dedup_removal_space(m, text, &repl(m)))
}

fn attr(m: &Captures, name: &str) -> String {
    // `dict(_EXPR_ATTR_RE.findall(attrs))` -- a repeated key keeps the *last* occurrence
    let attrs = m.name("attrs").map_or("", |a| a.as_str());
    EXPR_ATTR_RE
        .captures_iter(attrs)
        .map(|c| c.expect("unlimited backtracking"))
        .filter(|c| &c[1] == name)
        .last()
        .map_or_else(String::new, |c| c[2].to_string())
}

/// In document order. A prosody wrapper's inner words stay; only the delimiters go.
pub(crate) fn split_expr(text: &str) -> (String, Vec<ExpressiveTag>) {
    if !text.contains("<expr") && !text.contains("</expr") {
        return (text.to_string(), Vec::new());
    }
    let mut tags = Vec::new();
    let clean = sub_expr(&EXPR_OPEN_RE, text, |m| {
        tags.push(ExpressiveTag::new(attr(m, "type"), attr(m, "label")));
        String::new()
    });
    let clean = sub_expr(&EXPR_CLOSE_RE, &clean, |_| String::new());
    (clean, tags)
}

/// An unsupported type is a hallucination: its marker is dropped, its words kept. Gemini's
/// `expression` marker is not text but the part's style, so it stays for the plugin to lift out.
fn convert_expr(provider: Provider, text: &str) -> String {
    if !text.contains("<expr") && !text.contains("</expr") {
        return text.to_string();
    }

    // An unsupported marker lowers to `""`; `dedup_removal_space` avoids a double space.
    let text = sub_expr(&EXPR_WRAP_RE, text, |m| wrapping_marker(provider, m));
    let text = sub_expr(&EXPR_SELF_RE, &text, |m| point_marker(provider, m));

    // A stray unpaired expr tag must never reach the TTS: drop the delimiters, keep the words.
    // Gemini's standing style markers would match the opening pattern; only closers go there.
    let text = if provider == Provider::Gemini {
        text
    } else {
        sub_expr(&EXPR_OPEN_RE, &text, |_| String::new())
    };
    sub_expr(&EXPR_CLOSE_RE, &text, |_| String::new())
}

/// The `type` of an `<expr>` marker; anything else is a hallucination.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MarkerType {
    Expression,
    Sound,
    Break,
    Prosody,
    Spell,
}

impl MarkerType {
    fn of(m: &Captures) -> Option<Self> {
        match attr(m, "type").as_str() {
            "expression" => Some(Self::Expression),
            "sound" => Some(Self::Sound),
            "break" => Some(Self::Break),
            "prosody" => Some(Self::Prosody),
            "spell" => Some(Self::Spell),
            _ => None,
        }
    }
}

/// The prosody label every provider with a stress control understands.
const EMPHASIS: &str = "emphasis";

/// A spell or prosody wrapper in `provider`'s syntax; its inner words always stay.
fn wrapping_marker(provider: Provider, m: &Captures) -> String {
    let label = py_strip(&attr(m, "label")).to_lowercase();
    let inner = &m["inner"];
    if MarkerType::of(m) == Some(MarkerType::Spell) {
        return if provider == Provider::Cartesia {
            format!("<spell>{inner}</spell>")
        } else {
            inner.to_string()
        };
    }
    // prosody: native wrapping tags exist only for xAI
    match provider {
        Provider::Xai => {
            let native = label.replace(' ', "-");
            if XAI_WRAPPING.contains(&native.as_str()) {
                format!("<{native}>{inner}</{native}>")
            } else {
                inner.to_string()
            }
        }
        // not advertised for Inworld; salvage a stray one as a delivery hint
        Provider::Inworld => format!("<expression value=\"{label}\"/>{inner}"),
        // wrapping form of the point controls: apply before the span
        Provider::Cartesia => format!("{}{inner}", cartesia_prosody(&label).unwrap_or("")),
        Provider::FishAudio if label == EMPHASIS => format!("<emphasis>{inner}</emphasis>"),
        // tone controls are prefix markers: `[whispering]` steers the words after it
        Provider::FishAudio if FISHAUDIO_TONES.contains(&label.as_str()) => {
            format!("[{label}] {inner}")
        }
        Provider::FishAudio => inner.to_string(),
        // capitalizing is Gemini's one in-text stress control
        Provider::Gemini if label == EMPHASIS => inner.to_uppercase(),
        Provider::Gemini => inner.to_string(),
    }
}

/// A self-closing marker in `provider`'s syntax. Gemini keeps its `expression` marker as is.
fn point_marker(provider: Provider, m: &Captures) -> String {
    let marker_type = MarkerType::of(m);
    let label = attr(m, "label");
    if provider == Provider::Gemini && marker_type == Some(MarkerType::Expression) {
        let pre = m.name("pre").map_or(0, |p| p.as_str().len());
        return m[0][pre..].to_string();
    }
    match marker_type {
        Some(MarkerType::Expression) => expression_marker(provider, &label),
        Some(MarkerType::Sound) => sound_marker(provider, label),
        Some(MarkerType::Break) if provider == Provider::Gemini => format!("<{GEMINI_PAUSE}>"),
        Some(MarkerType::Break) => format!("<break time=\"{label}\"/>"),
        // Cartesia prosody is a self-closing point control (speed/volume)
        Some(MarkerType::Prosody) if provider == Provider::Cartesia => {
            cartesia_prosody(&py_strip(&label).to_lowercase()).unwrap_or("").to_string()
        }
        // Fish's native form is a prefix marker: keep a self-closing one as-is.
        Some(MarkerType::Prosody) if provider == Provider::FishAudio => {
            let tone = py_strip(&label).to_lowercase();
            if FISHAUDIO_TONES.contains(&tone.as_str()) {
                format!("[{tone}]")
            } else {
                String::new()
            }
        }
        _ => String::new(),
    }
}

fn expression_marker(provider: Provider, label: &str) -> String {
    match provider {
        // Cartesia's discrete emotion vocabulary (instructions list it)
        Provider::Cartesia => format!("<emotion value=\"{label}\"/>"),
        Provider::Inworld | Provider::FishAudio => format!("<expression value=\"{label}\"/>"),
        // xAI has no free-form delivery descriptions
        Provider::Xai | Provider::Gemini => String::new(),
    }
}

fn sound_marker(provider: Provider, label: String) -> String {
    match provider {
        // no non-verbal sound support
        Provider::Cartesia => String::new(),
        Provider::Xai => {
            let l = xai_sound_alias(&label.to_lowercase()).map_or(label, str::to_string);
            format!("<sound value=\"{l}\"/>")
        }
        Provider::FishAudio => {
            let l = fishaudio_sound_alias(&label.to_lowercase()).map_or(label, str::to_string);
            format!("<sound value=\"{l}\"/>")
        }
        Provider::Inworld => format!("<sound value=\"{label}\"/>"),
        Provider::Gemini => {
            let l = py_strip(&label).to_lowercase();
            let l = gemini_sound_alias(&l).unwrap_or(&l);
            if GEMINI_SOUNDS.contains(&l) { format!("<{l}>") } else { String::new() }
        }
    }
}

// --- native-tag rewrites ---------------------------------------------------------------------

/// `_xai_break_to_bracket` / `_fishaudio_break_to_bracket`, which differ only in the bracket names.
fn break_to_bracket(raw: &str, short: &str, long: &str) -> String {
    (if break_seconds(raw) >= 1.0 { long } else { short }).to_string()
}

/// The duration in a `<break time="...">`, `0.0` when it does not parse. `rstrip("s")` strips every
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

fn fishaudio_expression_to_bracket(value: &str) -> String {
    // Prefix "very" (steers Fish harder), never doubled when the LLM already wrote "very sad".
    let value = py_strip(value);
    if !value.is_empty() && !value.to_lowercase().starts_with("very ") {
        format!("[very {value}]")
    } else {
        format!("[{value}]")
    }
}

/// Fix common LLM markup mistakes for `provider`: close opening tags that should be
/// self-closing, or [`convert_markup`] silently deletes the cue (#6347).
pub(crate) fn normalize_markup(provider: Option<Provider>, text: &str) -> String {
    let Some(provider) = provider else { return text.to_string() };
    let text = sub(&EXPR_UNCLOSED_RE, text, |m| format!("{}/>", &m[1]));
    close_unclosed_native(provider, &text)
}

/// `re.sub(rf"<({pattern})\b([^>]*[^/])\s*>", r"<\1\2/>", text)` over `_SELF_CLOSING_TAGS`.
/// `[^/]` admits `>`, so `<break>  >` becomes `<break>/>`, as in CPython.
fn close_unclosed_native(provider: Provider, text: &str) -> String {
    fn pattern(provider: Provider) -> Regex {
        let tags = provider.self_closing_tags().join("|");
        compile(&format!(r"<({tags}){B}([^>]*[^/])[\s\x1c-\x1f]*>"))
    }
    static CARTESIA: LazyLock<Regex> = LazyLock::new(|| pattern(Provider::Cartesia));
    static INWORLD: LazyLock<Regex> = LazyLock::new(|| pattern(Provider::Inworld));
    static FISHAUDIO: LazyLock<Regex> = LazyLock::new(|| pattern(Provider::FishAudio));
    let re = match provider {
        Provider::Cartesia => &CARTESIA,
        Provider::Inworld => &INWORLD,
        Provider::FishAudio => &FISHAUDIO,
        Provider::Xai | Provider::Gemini => return text.to_string(),
    };
    sub(re, text, |m| format!("<{}{}/>", &m[1], &m[2]))
}

pub(crate) fn convert_markup(provider: Option<Provider>, text: &str) -> String {
    let Some(provider) = provider else { return text.to_string() };
    // Lower expr markers first; the conversions below handle the intermediate tags.
    let mut text = convert_expr(provider, text);
    if matches!(provider, Provider::Inworld | Provider::Xai) {
        text = convert_expression_tags(&text);
    }
    if provider == Provider::Xai {
        // xAI has no `<break>`; map it to its native `[pause]`/`[long-pause]`
        text = sub(&BREAK_RE, &text, |m| break_to_bracket(&m[1], "[pause]", "[long-pause]"));
    }
    if provider == Provider::FishAudio {
        // `_FISHAUDIO_EXPRESSION_RE` is `_EXPRESSION_RE`, character for character.
        text = sub(&EXPRESSION_RE, &text, |m| fishaudio_expression_to_bracket(&m[1]));
        text = convert_expression_tags(&text);
        text = sub(&BREAK_RE, &text, |m| break_to_bracket(&m[1], "[break]", "[long-break]"));
        text = sub(&FISHAUDIO_EMPHASIS_RE, &text, |m| format!("[emphasis] {}", py_strip(&m[1])));
    }
    // `<break>` is otherwise passed through unchanged: Inworld accepts it as native SSML.
    text
}
