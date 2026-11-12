//! Generated replies over every construct, each with what a listener, a reader and the history
//! must get from it, checked against every view under every chunking and every cut.

use super::*;
use crate::tts::markup::{Provider, ProviderExt};

const SPOKEN: View = View::Spoken { keep_markdown: false, keep_emoji: false };
const DISPLAY: View = View::Display { keep_emoji: true };
const VIEWS: [View; 3] = [SPOKEN, DISPLAY, View::Plain];

const PROVIDERS: [Option<Provider>; 7] = [
    None,
    Some(Provider::Cartesia),
    Some(Provider::Inworld),
    Some(Provider::Xai),
    Some(Provider::FishAudio),
    Some(Provider::FishAudioDrama),
    Some(Provider::Gemini),
];

/// Prose no rule may touch, including the near misses of every rule.
const PROSE: [&str; 22] = [
    "Hello",
    "there,",
    "how",
    "are",
    "you?",
    "café",
    "3 < 5",
    "a<b",
    "<3",
    "x > y",
    "<b>bold</b>",
    "5 * 3",
    "2*3*4",
    "snake_case",
    "#hashtag",
    "a [b] c",
    "50%",
    "C#",
    "1.",
    "a-b",
    "x~y",
    "it's",
];

/// A reply and its views, built node by node.
#[derive(Default)]
struct Doc {
    text: String,
    spoken: String,
    shown: String,
    plain: String,
}

impl Doc {
    fn add(&mut self, text: &str, spoken: &str, shown: &str, plain: &str) {
        self.text.push_str(text);
        self.spoken.push_str(spoken);
        self.shown.push_str(shown);
        self.plain.push_str(plain);
    }

    fn word(&mut self, w: &str) {
        self.add(w, w, w, w);
    }

    fn space(&mut self) {
        self.add(" ", " ", " ", " ");
    }

    fn view(&self, view: View) -> &str {
        match view {
            View::Spoken { .. } => &self.spoken,
            View::Display { .. } => &self.shown,
            _ => &self.plain,
        }
    }
}

fn pick<'a>(rng: &mut fastrand::Rng, from: &[&'a str]) -> &'a str {
    from[rng.usize(0..from.len())]
}

fn marker(rng: &mut fastrand::Rng, provider: Option<Provider>) -> String {
    let natives = provider.map(|p| p.native_tags()).unwrap_or(&[]);
    if !natives.is_empty() && rng.usize(0..3) == 0 {
        let name = pick(rng, natives);
        return match rng.usize(0..3) {
            0 => format!("<{name}/>"),
            1 => format!("<{name} value=\"x\"/>"),
            _ => format!("<{name}>"),
        };
    }
    let (kind, label) = match rng.usize(0..3) {
        0 => ("expression", pick(rng, &["warm", "calm > angry", "it's fine", "a*b", "[x]"])),
        1 => ("sound", pick(rng, &["laugh", "sigh", "clear throat"])),
        _ => ("break", pick(rng, &["1s", "300ms"])),
    };
    let q = if rng.bool() { '"' } else { '\'' };
    let name = pick(rng, &["expr", "EXPR", "Expr"]);
    let slash = pick(rng, &["/", " /", ""]);
    format!("<{name} type={q}{kind}{q} label={q}{label}{q}{slash}>")
}

/// One inline node, then a blank.
fn inline_node(rng: &mut fastrand::Rng, provider: Option<Provider>, d: &mut Doc) {
    let w = |rng: &mut fastrand::Rng| pick(rng, &PROSE[..6]);
    match rng.usize(0..10) {
        0 => {
            let m = marker(rng, provider);
            d.add(&m, "", "", "");
        }
        1 => {
            let (open, inner) = (pick(rng, &["*", "**", "***", "_", "__"]), w(rng));
            let raw = format!("{open}{inner}{open}");
            d.add(&raw, inner, &raw, inner);
        }
        2 => {
            let inner = w(rng);
            let raw = format!("~~{inner}~~");
            d.add(&raw, "", &raw, "");
        }
        3 => {
            // code is literal, so it quotes no markup here; that case has its own test
            let inner = pick(rng, &["x*y", "a_b", "ls -la", "a<b"]);
            let raw = format!("`{inner}`");
            d.add(&raw, inner, &raw, inner);
        }
        4 => {
            let (text, url) = (w(rng), pick(rng, &["https://x.y/a_b", "u"]));
            let raw = format!("[{text}]({url})");
            d.add(&raw, text, &raw, &format!("{text} ({url})"));
        }
        5 => {
            let e = pick(rng, &["\u{1F680}", "\u{2B50}\u{FE0F}", "\u{1F468}\u{200D}\u{1F469}"]);
            d.add(e, "", e, e);
        }
        _ => d.word(pick(rng, &PROSE)),
    }
    d.space();
}

fn doc(rng: &mut fastrand::Rng, provider: Option<Provider>) -> Doc {
    let mut d = Doc::default();
    for line in 0..rng.usize(1..5) {
        if line > 0 {
            d.add("\n", "\n", "\n", "\n");
        }
        match rng.usize(0..8) {
            0 => {
                let rule = pick(rng, &["---", "***", "- - -", "___"]);
                d.add(rule, "", rule, "");
                continue;
            }
            1 => {
                let code = pick(rng, &["print(1)", "a **b** 2*3", "x"]);
                let raw = format!("```py\n{code}\n```");
                d.add(&raw, "", &raw, &format!("{code}\n"));
                continue;
            }
            2..=4 => {
                let prefix = pick(rng, &["## ", "- ", "* ", "> ", "+ "]);
                d.add(prefix, "", prefix, "");
            }
            _ => {}
        }
        for _ in 0..rng.usize(1..6) {
            inline_node(rng, provider, &mut d);
        }
    }
    d
}

fn rules(provider: Option<Provider>) -> Rules {
    Rules::speech(Vocabulary::of(provider))
}

fn streamed(rules: &Rules, view: View, deltas: &[&str]) -> String {
    let mut lexer = Lexer::new(rules.clone());
    let mut p = Projector::new(view);
    let mut out = String::new();
    for d in deltas {
        out.push_str(&p.text(&lexer.push(d)));
        assert!(lexer.held() <= MAX_HELD, "held {} bytes after {d:?}", lexer.held());
    }
    out.push_str(&p.text(&lexer.finish()));
    out
}

/// The most any recognizer may hold: a tag or an inline window, plus a character.
const MAX_HELD: usize = markup::MAX_TAG + INLINE_WINDOW + 4;

fn words(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}

fn random_split<'a>(rng: &mut fastrand::Rng, text: &'a str) -> Vec<&'a str> {
    let mut cuts: Vec<usize> = (0..rng.usize(0..10))
        .map(|_| rng.usize(0..=text.len()))
        .filter(|i| text.is_char_boundary(*i))
        .collect();
    cuts.extend([0, text.len()]);
    cuts.sort_unstable();
    cuts.dedup();
    cuts.windows(2).map(|w| &text[w[0]..w[1]]).collect()
}

fn assert_nothing_leaks(text: &str, ctx: &str) {
    let lower = text.to_lowercase();
    for fragment in ["<expr", "</expr", "label=", "type=", "<think", "</think"] {
        assert!(!lower.contains(fragment), "{ctx}: {fragment:?} in {text:?}");
    }
}

#[test]
fn every_view_of_a_generated_reply_is_its_ground_truth() {
    let mut rng = fastrand::Rng::with_seed(0x5eed);
    for case in 0..4000 {
        let provider = PROVIDERS[case % PROVIDERS.len()];
        let rules = rules(provider);
        let d = doc(&mut rng, provider);
        let tokens = Lexer::all(rules.clone(), &d.text);
        let deltas = random_split(&mut rng, &d.text);
        for view in VIEWS {
            let ctx = format!("case {case} {provider:?} {view:?} {:?}", d.text);
            let whole = Projector::new(view).text(&tokens);
            assert_eq!(words(&whole), words(d.view(view)), "{ctx}");
            assert_nothing_leaks(&whole, &ctx);
            // however the stream is chunked, the same text comes out
            assert_eq!(streamed(&rules, view, &deltas), whole, "{ctx} as {deltas:?}");
        }
        // no view carries markup or reasoning a second reading would find (code is literal, so
        // a view is not idempotent: `` `**b**` `` is spoken `**b**`)
        for view in VIEWS {
            let once = Projector::new(view).text(&tokens);
            let again =
                Lexer::all(Rules { model: Some(ModelSyntax::DEFAULT), ..rules.clone() }, &once);
            assert!(
                again.iter().all(|t| !matches!(t.kind, Kind::Tag(_) | Kind::Hidden(_))),
                "case {case} {view:?} {once:?}"
            );
        }
    }
}

#[test]
fn one_character_at_a_time_is_the_same_as_all_at_once() {
    let mut rng = fastrand::Rng::with_seed(0xc4a7);
    for case in 0..600 {
        let provider = PROVIDERS[case % PROVIDERS.len()];
        let rules = rules(provider);
        let d = doc(&mut rng, provider);
        let chars: Vec<String> = d.text.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        for view in VIEWS {
            assert_eq!(
                streamed(&rules, view, &chars),
                streamed(&rules, view, &[&d.text]),
                "case {case} {view:?} {:?}",
                d.text
            );
        }
    }
}

#[test]
fn a_reply_cut_anywhere_shows_only_what_it_had_and_no_markup() {
    let mut rng = fastrand::Rng::with_seed(0xc07);
    for case in 0..250 {
        let provider = PROVIDERS[case % PROVIDERS.len()];
        let rules = rules(provider);
        let d = doc(&mut rng, provider);
        let shown_words = words(&d.shown);
        for (cut, _) in d.text.char_indices() {
            let prefix = &d.text[..cut];
            for view in VIEWS {
                let got = streamed(&rules, view, &[prefix]);
                let ctx = format!("case {case} {provider:?} {view:?} cut {prefix:?}");
                assert_nothing_leaks(&got, &ctx);
                // every word came from the reply, in order (a link's text and url from one word);
                // the last may be cut short
                let bare =
                    |s: &str| s.trim_matches(|c: char| "*_~`[]()!#>-+".contains(c)).to_owned();
                let got_words = words(&got);
                let mut at = 0;
                for w in &got_words {
                    let found = shown_words[at..].iter().position(|s| bare(s).contains(&bare(w)));
                    let Some(found) = found else {
                        panic!("{ctx}: {w:?} of {got:?} not in {shown_words:?} from {at}");
                    };
                    at += found;
                }
            }
        }
    }
}

fn units(rules: &Rules, deltas: &[&str]) -> Vec<Unit> {
    let mut lexer = Lexer::new(rules.clone());
    let mut sync = Sync::new(SPOKEN, DISPLAY);
    let mut out = Vec::new();
    for d in deltas {
        out.extend(sync.push(&lexer.push(d)));
    }
    out.extend(sync.push(&lexer.finish()));
    out
}

#[test]
fn the_synchronizer_shows_exactly_the_transcript_and_paces_exactly_the_spoken_words() {
    let mut rng = fastrand::Rng::with_seed(0x5c);
    for case in 0..3000 {
        let provider = PROVIDERS[case % PROVIDERS.len()];
        let rules = rules(provider);
        let d = doc(&mut rng, provider);
        let tokens = Lexer::all(rules.clone(), &d.text);
        let shown = Projector::new(DISPLAY).text(&tokens);
        let said = Projector::new(SPOKEN).text(&tokens);
        for deltas in [vec![d.text.as_str()], random_split(&mut rng, &d.text)] {
            let ctx = format!("case {case} {provider:?} {:?} as {deltas:?}", d.text);
            let units = units(&rules, &deltas);
            let joined: String = units.iter().map(Unit::shown).collect();
            assert_eq!(joined, shown, "{ctx}");
            let paced: String = units
                .iter()
                .filter_map(|u| match u {
                    Unit::Said(text) => Some(text.as_str()),
                    Unit::Silent { .. } => None,
                })
                .collect::<Vec<_>>()
                .concat();
            assert_eq!(words(&paced), words(&said), "{ctx}");
        }
    }
}

/// Text built to sit on every window's edge: long runs of each special character, inside and
/// around each construct's opener.
fn edge_text(rng: &mut fastrand::Rng) -> String {
    const OPENERS: [&str; 16] = [
        "<expr ",
        "<expr label=\"",
        "<expr label=",
        "<expr type",
        "</expr",
        "<",
        "**",
        "~~",
        "`",
        "```",
        "[a](",
        "[",
        "\n",
        "x ",
        "<think>",
        "</think>",
    ];
    const RUNS: [&str; 16] =
        ["*", "_", "~", "`", "#", "-", " ", "\t", "\n", "x", "é", ">", "<", "\"", "🎉", "\u{200D}"];
    // around the inline window and the tag limit
    const LENGTHS: [usize; 15] =
        [1, 2, 3, 4, 150, 158, 159, 160, 161, 162, 254, 255, 256, 257, 600];
    let mut text = String::new();
    for _ in 0..rng.usize(1..5) {
        if rng.bool() {
            text.push_str(pick(rng, &OPENERS));
        } else {
            let len = LENGTHS[rng.usize(0..LENGTHS.len())];
            text.push_str(&pick(rng, &RUNS).repeat(len));
        }
    }
    text
}

#[test]
fn text_on_a_windows_edge_is_held_within_it_and_reads_alike_however_it_arrives() {
    let mut rng = fastrand::Rng::with_seed(0xed9e);
    let all = Rules { model: Some(ModelSyntax::DEFAULT), ..Rules::speech(Vocabulary::All) };
    for case in 0..1500 {
        let rules = if case % 2 == 0 { all.clone() } else { rules(Some(Provider::Cartesia)) };
        let text = edge_text(&mut rng);
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        for view in [SPOKEN, DISPLAY, View::Plain, View::Model] {
            let whole = streamed(&rules, view, &[&text]);
            let ctx = format!("case {case} {view:?} {text:?}");
            assert_eq!(streamed(&rules, view, &chars), whole, "{ctx} one character at a time");
            let deltas = random_split(&mut rng, &text);
            assert_eq!(streamed(&rules, view, &deltas), whole, "{ctx} as {deltas:?}");
        }
    }
}
