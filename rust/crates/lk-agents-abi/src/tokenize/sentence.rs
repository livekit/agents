//! `tokenize/_basic_sent.py::split_sentences`: the rule-based splitter behind `basic.SentenceTokenizer`
//! (https://stackoverflow.com/a/31505798). The text may not contain `<prd>` or `<stop>`. Offsets
//! are characters.

use std::sync::LazyLock;

use regex::Regex;

use crate::tokenize::py_strip;

const STARTERS: &str = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)";
const SUFFIXES: &str = "(Inc|Ltd|Jr|Sr|Co)";

fn rule(pattern: &str, rep: &'static str) -> (Regex, &'static str) {
    (Regex::new(pattern).expect("constant pattern"), rep)
}

static PREFIXES: LazyLock<[(Regex, &str); 2]> = LazyLock::new(|| {
    [rule(r"(Mr|St|Mrs|Ms|Dr)[.]", "${1}<prd>"), rule(r"[.](com|net|org|io|gov|edu|me)", "<prd>${1}")]
});

static MULTIPLE_DOTS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\.{2,}").expect("constant pattern"));

static RULES: LazyLock<Vec<(Regex, &str)>> = LazyLock::new(|| {
    vec![
        rule(r"\s([A-Za-z])[.] ", " ${1}<prd> "),
        rule(&format!(r"([A-Z][.][A-Z][.](?:[A-Z][.])?) {STARTERS}"), "${1}<stop> ${2}"),
        rule(r"([A-Za-z])[.]([A-Za-z])[.]([A-Za-z])[.]", "${1}<prd>${2}<prd>${3}<prd>"),
        rule(r"([A-Za-z])[.]([A-Za-z])[.]", "${1}<prd>${2}<prd>"),
        rule(&format!(r" {SUFFIXES}[.] {STARTERS}"), " ${1}<stop> ${2}"),
        rule(&format!(r" {SUFFIXES}[.]"), " ${1}<prd>"),
        rule(r" ([A-Za-z])[.]", " ${1}<prd>"),
        rule(r#"([.!?。！？])(["”])"#, "${1}${2}<stop>"),
    ]
});

/// Sentences of more than `min_sentence_len` characters (shorter ones join the next), each with
/// its span in the text.
#[crate::host(export(pid = false, kw_only = "retain_format", defaults(min_sentence_len = 20, retain_format = false)))]
pub fn basic_split_sentences(
    text: &str,
    min_sentence_len: usize,
    retain_format: bool,
) -> Vec<(String, usize, usize)> {
    let mut text = text.replace('\n', if retain_format { "<nel><stop>" } else { " " });
    for (re, rep) in PREFIXES.iter() {
        text = re.replace_all(&text, *rep).into_owned();
    }
    // A number can hold several dot-separated groups ("192.168.1.1").
    text = protect_digit_dots(&text);
    text = MULTIPLE_DOTS.replace_all(&text, |c: &regex::Captures| "<prd>".repeat(c[0].len())).into_owned();
    if text.contains("Ph.D") {
        text = text.replace("Ph.D.", "Ph<prd>D<prd>");
    }
    for (re, rep) in RULES.iter() {
        text = re.replace_all(&text, *rep).into_owned();
    }
    let mut text = mark_stops(&text).replace("<prd>", ".");
    if retain_format {
        text = text.replace("<nel>", "\n");
    }
    let total = text.replace("<stop>", "").chars().count();

    let pre_pad = if retain_format { "" } else { " " };
    let mut sentences = Vec::new();
    let mut buff = String::new();
    let (mut start, mut end) = (0, 0);
    for piece in text.split("<stop>") {
        let sentence = if retain_format { piece } else { py_strip(piece) };
        if sentence.is_empty() {
            continue;
        }
        buff.push_str(pre_pad);
        buff.push_str(sentence);
        end += piece.chars().count();
        if buff.chars().count() > min_sentence_len {
            sentences.push((buff[pre_pad.len()..].to_owned(), start, end));
            start = end;
            buff.clear();
        }
    }
    if !buff.is_empty() {
        sentences.push((buff[pre_pad.len()..].to_owned(), start, total));
    }
    sentences
}

/// `(?<=[0-9])\.(?=[0-9])` -> `<prd>`.
fn protect_digit_dots(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let digit = |i: usize| chars.get(i).is_some_and(char::is_ascii_digit);
    let mut out = String::with_capacity(text.len());
    for (i, c) in chars.iter().enumerate() {
        if *c == '.' && i > 0 && digit(i - 1) && digit(i + 1) {
            out.push_str("<prd>");
        } else {
            out.push(*c);
        }
    }
    out
}

/// `([.!?。！？])(?!["”])` -> `\1<stop>`.
fn mark_stops(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(c);
        if matches!(c, '.' | '!' | '?' | '。' | '！' | '？') && !matches!(chars.peek(), Some('"' | '”')) {
            out.push_str("<stop>");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str, min: usize, retain: bool) -> Vec<(String, usize, usize)> {
        basic_split_sentences(text, min, retain)
    }

    fn owned(want: &[(&str, usize, usize)]) -> Vec<(String, usize, usize)> {
        want.iter().map(|(t, s, e)| ((*t).to_owned(), *s, *e)).collect()
    }

    // Every expectation is the reference's output.
    #[test]
    fn sentences_match_the_reference() {
        for (text, min, retain, want) in [
            (
                "Mr. Smith paid 1.2.3 dollars at acme.com today... It was fine. Dr. Who? Yes!",
                1,
                false,
                &[
                    ("Mr. Smith paid 1.2.3 dollars at acme.com today... It was fine.", 0, 62),
                    ("Dr. Who?", 62, 71),
                    ("Yes!", 71, 76),
                ][..],
            ),
            (
                "Hi. Héllo there, how are you? Fine.",
                20,
                false,
                &[("Hi. Héllo there, how are you?", 0, 29), ("Fine.", 29, 35)],
            ),
            (
                "He said \"stop.\" Then left. 你好。再见！",
                1,
                false,
                &[("He said \"stop.\"", 0, 15), ("Then left.", 15, 26), ("你好。", 26, 30), ("再见！", 30, 33)],
            ),
            (
                "First line here.\nSecond one here.",
                1,
                true,
                &[("First line here.", 0, 16), ("\nSecond one here.", 16, 33)],
            ),
            (
                "I work at Acme Inc. He said hi. U.S.A. is big. The U.S. They went. e.g. this.",
                1,
                false,
                &[
                    ("I work at Acme Inc", 0, 18),
                    ("He said hi.", 18, 30),
                    ("U.S.A. is big.", 30, 45),
                    ("The U.S.", 45, 54),
                    ("They went.", 54, 65),
                    ("e.g. this.", 65, 76),
                ],
            ),
        ] {
            assert_eq!(split(text, min, retain), owned(want), "{text:?}");
        }
    }
}
