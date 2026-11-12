//! BlingFire's sentence rules (`ldbsrc/sbd/wbd.lex.utf8` at github.com/microsoft/BlingFire
//! 18e9a19, MIT, see `fixtures/sentences/LICENSE`) as regexes,
//! with the changes speech needs marked `Voice:`.
//!
//! A rule's pattern includes its context; `left` and `right` count the context symbols on each
//! side of the span the rule acts on. `^` and `$` are the [`START`] and [`END`] symbols.

use std::sync::LazyLock;

use regex_automata::MatchKind;
use regex_automata::hybrid::dfa::DFA;

/// `^`: the symbol before the text.
pub(super) const START: char = '\u{FDD0}';
/// `$`: the symbol after the text.
pub(super) const END: char = '\u{FDD1}';
/// A codepoint in no class, standing in for input that would collide with the anchors.
pub(super) const OTHER: char = '\u{FDD2}';

#[derive(Debug, Clone, Copy)]
pub(super) enum Action {
    /// The span ends a sentence.
    Eos,
    /// The span's first symbol ends a sentence; the rest of the span is consumed.
    EosAtFirst,
    /// The span is handed to another rule set; `once` stops that set after its first match.
    Call { set: Set, once: bool },
    /// The span is consumed without a sentence end.
    NoBreak,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Set {
    Main,
    LastDot,
    MarkTerminal,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Rule {
    pub left: usize,
    pub right: usize,
    pub action: Action,
}

pub(super) struct RuleSet {
    pub dfa: DFA,
    pub rules: Vec<Rule>,
}

/// The three rule sets, indexed by [`Set`].
pub(super) type RuleSets = [RuleSet; 3];

/// The rule sets compiled from [`Sources::default`].
pub(super) fn rule_sets() -> &'static RuleSets {
    static SETS: LazyLock<RuleSets> = LazyLock::new(|| Patterns::new(&Sources::default()).sets());
    &SETS
}

/// What the rules are built from: the word lists and the character sets a rule names.
#[derive(Debug, Clone)]
pub(super) struct Sources {
    pub abbreviations: Vec<&'static str>,
    pub titles: Vec<&'static str>,
    /// Common words left out of the abbreviations, still read as one before a number.
    pub not_abbreviations: Vec<&'static str>,
    pub start_words: Vec<&'static str>,
    pub ordinal_nouns: Vec<&'static str>,
    pub always: &'static str,
    pub endp: &'static str,
    pub closers: &'static str,
    pub emoji: &'static str,
    pub capital: &'static str,
}

impl Default for Sources {
    fn default() -> Self {
        Sources {
            abbreviations: words(include_str!("data/en_abbreviations.txt"))
                .chain(words(include_str!("data/more_abbreviations.txt")))
                .collect(),
            titles: words(include_str!("data/en_titles.txt"))
                .chain(words(include_str!("data/more_titles.txt")))
                .collect(),
            not_abbreviations: words(include_str!("data/not_abbreviations.txt")).collect(),
            start_words: words(include_str!("data/en_start_words.txt")).collect(),
            ordinal_nouns: ORDINAL_NOUNS.split_whitespace().collect(),
            always: ALWAYS,
            endp: ENDP,
            closers: CLOSERS,
            emoji: EMOJI,
            capital: CAPITAL,
        }
    }
}

pub(super) fn build(rules: Vec<(String, Rule)>) -> RuleSet {
    let patterns: Vec<&str> = rules.iter().map(|(p, _)| p.as_str()).collect();
    let dfa = DFA::builder()
        .configure(
            DFA::config()
                .match_kind(MatchKind::All)
                .cache_capacity(16 << 20)
                .minimum_cache_clear_count(None),
        )
        .build_many(&patterns)
        .expect("the sentence rules compile");
    RuleSet { dfa, rules: rules.into_iter().map(|(_, r)| r).collect() }
}

fn words(list: &'static str) -> impl Iterator<Item = &'static str> {
    list.lines().filter(|w| !w.is_empty())
}

fn alternation<'a>(words: impl Iterator<Item = &'a str>) -> String {
    let words: Vec<String> = words.map(regex::escape).collect();
    format!("(?:{})", words.join("|"))
}

/// Punctuation, as the rules see it.
const PUNCTUATION: &str = r"\p{P}";
/// Whitespace, as the rules see it.
const WS: &str = r"\x{4}-\x{20}\x{7F}-\x{9F}\x{A0}\x{2000}-\x{200B}\x{200E}\x{200F}\x{202F}\x{205F}\x{2060}\x{2420}\x{2424}\x{3000}\x{FEFF}";
/// Terminators that end a sentence wherever they appear: ideographic and fullwidth marks, line
/// and paragraph separators. Voice: the Indic danda and double danda, and the Burmese, Armenian
/// and Ethiopic full stops.
const ALWAYS: &str =
    r"\x{3002}\x{FF1F}\x{FF01}\x{2028}\x{2029}\x{964}\x{965}\x{104B}\x{589}\x{1362}";
/// `[?.!...]`: terminators that end a sentence before a capital or the end of the text.
const ENDP: &str = r"!.?\x{1C3}\x{6D4}\x{61F}\x{2024}\x{2026}\x{203C}\x{203D}\x{2048}\x{2049}\x{2404}\x{FE52}\x{FF0E}\x{FF61}";
/// Voice: closing quotes and brackets, which belong to the sentence they close even after a
/// space (`« Bonjour ! »`).
const CLOSERS: &str = r"»”’」』）)\]》】〉›〕｣";
/// Voice: an emoji right after a sentence end belongs to that sentence (`Merci ! 🙂`).
const EMOJI: &str = r"[\p{Extended_Pictographic}\x{FE0F}\x{200D}\x{1F3FB}-\x{1F3FF}]";
/// What may start a sentence: an uppercase letter, or a letter of a script without case
/// (Chinese, Japanese, Devanagari, Arabic...).
const CAPITAL: &str = r"[\p{Lu}\p{Lt}\p{Lo}\p{Lm}]";
/// Voice: German nouns a day or ordinal number precedes (`am 3. Oktober`, `im 2. Stock`).
const ORDINAL_NOUNS: &str = "Januar Februar März April Mai Juni Juli August September Oktober \
    November Dezember Jan Feb Mär Apr Jun Jul Aug Sep Sept Okt Nov Dez Jahrhundert Jh Jhd \
    Stock Etage Platz Rang Mal Klasse Weltkrieg Geburtstag Lebensjahr Liga Runde Quartal \
    Kapitel Advent Woche Tag";

/// The building blocks shared by the rules, named as in the BlingFire source.
pub(super) struct Patterns {
    always: &'static str,
    closers: &'static str,
    emoji: &'static str,
    ordinal_noun: String,
    ws: String,
    px: String,
    pw: String,
    letter: String,
    number: String,
    capital: String,
    endp: String,
    term: String,
    tail: String,
    left: String,
    abbr_seq: String,
    abbr: String,
    abbr_before_number: String,
    title: String,
    start_word: String,
    bullet: String,
    reference: String,
}

impl Patterns {
    pub(super) fn new(src: &Sources) -> Self {
        let Sources { always, endp, closers, emoji, capital, .. } = *src;
        let ws = format!("[{WS}]");
        let px = format!("[{PUNCTUATION}]");
        let letter = r"[\p{L}\p{M}]".to_owned();
        let short = |n: usize| format!("{letter}{{1,{n}}}");
        let bullet_body =
            r"(?:[0-9][0-9]?|[a-z]|i?i?i|iv|vi?i?i?|ix|xi?i?i?|I?I?I|IV|VI?I?I?|IX|XI?I?I?)";
        let sp = r"[\x{20}\x{A0}]";
        let n3 = "[0-9]?[0-9]?[0-9]";
        let removed = &src.not_abbreviations;
        let all = || src.abbreviations.iter().chain(&src.titles).copied();
        let abbreviations = all().filter(|w| !removed.contains(w));
        let abbreviations_and_removed = all().chain(removed.iter().copied());
        Patterns {
            always,
            closers,
            emoji,
            ordinal_noun: alternation(src.ordinal_nouns.iter().copied()),
            pw: format!("(?:{px}|{ws})"),
            left: format!(r"(?:\x{{FDD0}}|{px}|{ws})"),
            number: r"\p{N}".to_owned(),
            capital: capital.to_owned(),
            endp: format!("[{endp}]"),
            term: format!("[{endp}{always}]"),
            tail: format!(r#"(?:[{closers}"']|{ws}+[{closers}]|{ws}*{emoji}+)*"#),
            // Voice: spaced pieces are one or two letters (`z. B.`, `Ph. D.`, `EE. UU.`), so a
            // run of short replies (`gaya. Dr. Sharma`, `inkl. MwSt. Zum`) is not one abbreviation.
            abbr_seq: format!(
                "(?:(?:{l4}[.])+{l4}[.]|(?:{l2}[.]{ws})+{l2}[.])",
                l4 = short(4),
                l2 = short(2)
            ),
            abbr: format!("{}[.]", alternation(abbreviations)),
            // Voice: what reads as an abbreviation before a number: the list, the entries left
            // out of it (`no. 3`, `al. 2`) and single letters (`p. 87`).
            abbr_before_number: format!(
                "(?:{}|{letter})[.]",
                alternation(abbreviations_and_removed)
            ),
            title: format!("{}[.]", alternation(src.titles.iter().copied())),
            start_word: alternation(src.start_words.iter().copied()),
            bullet: format!(
                r"(?:{bullet_body}[ ]?[.)]|\[[ ]?{bullet_body}[ ]?\]|[(][ ]?{bullet_body}[ ]?[)])"
            ),
            reference: format!(
                r"(?:\[{sp}?{n3}{sp}?\]|{sp}?:{n3}(?:{sp}?[-,–]{sp}?{n3})?{sp}?|\[{sp}?[cC][iI][tT][aA][tT][iI][oO][nN]{sp}[nN][eE][eE][dD][eE][dD]{sp}?\]|\[{sp}?[nN][eE][eE][dD]{sp}[cC][iI][tT][aA][tT][iI][oO][nN]{sp}?\])"
            ),
            ws,
            px,
            letter,
        }
    }

    pub(super) fn sets(&self) -> RuleSets {
        [build(self.main()), self.last_dot(), self.mark_terminal()]
    }

    /// The top-level rules, in priority order.
    pub(super) fn main(&self) -> Vec<(String, Rule)> {
        let Patterns {
            always,
            closers: _,
            emoji,
            ordinal_noun,
            ws,
            px,
            pw,
            letter,
            number,
            capital,
            endp,
            term: _,
            tail,
            left,
            abbr_seq,
            abbr,
            abbr_before_number,
            title,
            start_word,
            bullet,
            reference,
        } = self;
        let rule = |pattern: String, left, right, action| (pattern, Rule { left, right, action });
        let mark = Action::Call { set: Set::MarkTerminal, once: false };
        // Only the first terminator of the span, so a list number after it is not one.
        let mark_first = Action::Call { set: Set::MarkTerminal, once: true };
        let last_dot = Action::Call { set: Set::LastDot, once: false };
        let no_break = Action::NoBreak;
        // Voice: the word or symbol run before a comma stops at a terminator, so `。` inside a
        // run of Chinese followed by `, ` still ends its sentence.
        let run = format!(r"(?:{letter}|{number}|[\p{{P}}--[{always}]])");
        let cjk = r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]";
        let thai = r"[\x{E01}-\x{E3A}\x{E40}-\x{E45}\x{E47}-\x{E4E}]";
        vec![
            // Voice: the closers and a run of the same terminator (`。。。`) stay with it.
            rule(format!("(?:[{always}]+{tail})+"), 0, 0, Action::Eos),
            // Voice: a quoted terminator is a word, not an end (`हिंदी में "।" ज़्यादा चलता है`).
            rule(format!(r#"["“'‘][{always}]+["”'’]"#), 0, 0, no_break),
            // Voice: a Japanese quotation goes on with its particle (`「承知しました。」と`).
            rule(format!("[{always}]+[」』》）)]+[とってをがはもにでのや]"), 0, 0, no_break),
            // Voice: a list number after it starts the next sentence (`套餐。1. 基础版`).
            rule(format!("[{always}]+{tail}{ws}*{bullet}{ws}"), 0, 1, mark_first),
            rule(format!(r"{endp}{pw}*\x{{FDD1}}"), 0, 1, mark),
            rule(format!("{endp}{ws}+{pw}*{capital}"), 0, 1, mark),
            rule(format!("{endp}{pw}*(?:{ws}*{emoji}+)+{ws}+{capital}"), 0, 1, mark),
            rule(format!(r"{endp}{pw}*(?:{ws}*{emoji}+)+{ws}*\x{{FDD1}}"), 0, 1, mark),
            rule(format!("{endp}{pw}*{ws}{capital}"), 0, 2, mark),
            // 1. Sentence one." 2. Sentence two. Voice: the list number is never a sentence
            // of its own, and digits read one by one (`4. 2. 9.`) are not a list.
            rule(format!(r"{endp}{ws}+{pw}*{bullet}{ws}[^\d\s]"), 0, 2, mark_first),
            rule(format!(r"{endp}{pw}*{ws}+{bullet}{ws}[^\d\s]"), 0, 2, mark_first),
            // Voice: a list number at the start of the text, of a line, or after a colon or
            // a comma is not a sentence of its own.
            rule(format!(r"\x{{FDD0}}{ws}*{bullet}{ws}"), 1, 1, no_break),
            // Voice: a list item on a new line ends the line before it, whatever the item starts
            // with (`1. Router ausschalten.⏎2. 30 Sekunden warten.`), markdown bullets included.
            rule(
                format!("[^{WS}]{ws}*\n{ws}*(?:{bullet}|[-•*·‣▪–]){ws}"),
                0,
                1,
                Action::EosAtFirst,
            ),
            rule(format!("[、，：]{ws}*{bullet}{ws}"), 1, 1, no_break),
            rule(format!("[,]{ws}+{bullet}{ws}"), 1, 1, no_break),
            // Voice: after a colon only a first item starts a list (`Pasos: 1. Abre`), not a
            // value (`Comensales: 6. ¿Todo correcto?`).
            rule(format!(r"[:]{ws}+(?:1|a|i|I|A)[ ]?[.)]{ws}"), 1, 1, no_break),
            // Voice: digits read one by one (`4. 2. 9. 6.`) are one unit; only the last
            // period may end the sentence.
            rule(
                format!(r"{left}(?:\d{{1,4}}[.][ \x{{A0}}\x{{202F}}]+)+\d{{1,4}}[.]"),
                1,
                1,
                no_break,
            ),
            rule(format!(r"{endp}{ws}+[\x{{BF}}\x{{A1}}]"), 0, 0, mark),
            // A word followed by a comma never ends a sentence.
            rule(format!("{left}{run}+{ws}?,{ws}"), 1, 1, no_break),
            // C. D. Jr., e.g., t.p.
            rule(format!("{left}{abbr_seq}{pw}"), 1, 1, no_break),
            rule(format!("{left}{abbr}{pw}"), 1, 1, no_break),
            rule(format!("{left}[A-Z][.]{pw}"), 1, 1, no_break),
            // Voice: a title or initial right after a sentence end keeps its name
            // (`mil gaya. Dr. Sharma`).
            rule(format!("{endp}{px}*{ws}+(?:{title}|[A-Z][.]){ws}"), 0, 1, mark_first),
            // Voice: a short word before a number or a code is an abbreviation (`p. 87`,
            // `al. 2`, `no. 3`, `Reference no.: RM-2026`).
            rule(format!(r"{left}{abbr_before_number}{ws}+(?:[A-Z][.]{ws}*)?\d"), 1, 1, no_break),
            rule(format!(r"{left}[nN][oOrR][.]:?{ws}+[A-Z0-9#][A-Z0-9\-/]*\d"), 1, 1, no_break),
            // ... in Nov. The new ...
            rule(format!("{left}{abbr}{pw}+{start_word}{pw}"), 1, 1, last_dot),
            rule(format!("{left}{abbr_seq}{pw}+{start_word}{pw}"), 1, 1, last_dot),
            rule(format!("{left}[A-Z][.]{pw}+{start_word}{pw}"), 1, 1, last_dot),
            // Voice: a spelled-out word ends on its last letter (`N, G, U, Y, E, N. Would`).
            rule(format!(r"[A-Z0-9][,\-]{ws}?[A-Z][.]{ws}+{capital}"), 0, 1, mark_first),
            // Voice: German day and ordinal numbers (`den 25. Februar`, `im 2. Weltkrieg`).
            rule(format!("{left}[0-9][0-9]?[.]{ws}+{ordinal_noun}{pw}"), 1, 1, no_break),
            // Voice: sentences glued together, as after a tool call (`name?What`, `you.Your`).
            rule(r"[\p{Ll}\d][?!\x{203D}\x{2047}-\x{2049}]+\p{Lu}\p{Ll}".to_owned(), 1, 2, mark),
            rule(r"[\p{Ll}\d][.]\p{Lu}\p{Ll}".to_owned(), 1, 2, mark),
            rule(format!("{cjk}(?:[!?]+|[.]){cjk}"), 1, 1, mark),
            // Voice: Thai ends a sentence or clause with a space between Thai words.
            rule(format!("{thai}{ws}{thai}"), 0, 2, Action::Eos),
            // Like this.[1] Or like this.":64  Instead ...
            rule(format!(r#"{endp}"?{ws}?{reference}{ws}{pw}*{capital}"#), 0, 1, Action::Eos),
            // ... bbb."
            rule(format!(r#"{endp}"(?:{ws}*{endp})?{ws}"#), 0, 1, Action::Eos),
            // Scribner. pp. 352-366
            rule(format!("{endp}{ws}+[pP][pP]?[.]{ws}?[0-9][0-9]?[0-9]?{pw}"), 0, 1, no_break),
        ]
    }

    /// The last dot of a span ends the sentence.
    pub(super) fn last_dot(&self) -> RuleSet {
        build(vec![(
            r"[.][^.\x{FDD0}\x{FDD1}]+\x{FDD1}".to_owned(),
            Rule {
                left: 0,
                right: 1,
                action: Action::Call { set: Set::MarkTerminal, once: false },
            },
        )])
    }

    /// Terminal punctuation and the punctuation attached to it, e.g. `."`. Voice: a closing
    /// quote or bracket after a space too (`demain. »`).
    pub(super) fn mark_terminal(&self) -> RuleSet {
        let Patterns { ws, px, term, closers, emoji, .. } = self;
        build(vec![(
            format!("(?:{ws}*{term}+(?:{px}|{ws}+[{closers}]|{ws}*{emoji}+)*)+"),
            Rule { left: 0, right: 0, action: Action::Eos },
        )])
    }
}
