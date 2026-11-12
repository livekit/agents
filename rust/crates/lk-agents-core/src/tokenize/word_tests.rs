use super::*;

const DEFAULTS: WordTokenizerOptions =
    WordTokenizerOptions { ignore_punctuation: true, retain_format: false };
const KEEP_PUNCTUATION: WordTokenizerOptions =
    WordTokenizerOptions { ignore_punctuation: false, retain_format: false };
const RETAIN: WordTokenizerOptions =
    WordTokenizerOptions { ignore_punctuation: false, retain_format: true };

/// `(text, start, end)` triples, for array-literal expectations.
fn spans(text: &str, opts: WordTokenizerOptions) -> Vec<(String, usize, usize)> {
    split_words(text, opts).into_iter().map(|w| (w.text.into_owned(), w.start, w.end)).collect()
}

fn texts(text: &str, opts: WordTokenizerOptions) -> Vec<String> {
    split_words(text, opts).into_iter().map(|w| w.text.into_owned()).collect()
}

/// Every language below, for the properties that hold over all of them.
const SAMPLES: &[&str] = &[
    "Hello, world! It's 3.5 o'clock.",
    "C'est l'été, déjà !",
    "¿Qué tal? ¡Muy bien!",
    "Größe: 5 Äpfel.",
    "我用LiveKit开发。",
    "這是繁體中文，𠮷野家。",
    "東京タワーへ行きます",
    "ｶﾞｲﾄﾞ と か\u{3099}",
    "「はい」、人々。",
    "क्षत्रिय नमस्ते, क़िला।",
    "Kal meeting hai, theek है?",
    "안녕하세요, 반갑습니다!",
    "สวัสดีครับ ยินดีต้อนรับ",
    "👍🏽 ok 👨\u{200d}👩\u{200d}👧 好👍🏽了",
    "  a  b ",
    "a\u{1e}b...",
    "a 世界",
    "!!!",
    "   ",
    "",
];

// Catches: `ignore_punctuation` as an edge trim (`str.translate` deletes everywhere; "don't"
// and "dont" must match in `_transcripts_equivalent`), or spans of the stripped text.
#[test]
fn english_punctuation_is_deleted_inside_the_word_and_offsets_span_the_source() {
    assert_eq!(
        spans("Hello, world! It's 3.5 o'clock.", DEFAULTS),
        [
            ("Hello".into(), 0, 6),
            ("world".into(), 7, 13),
            ("Its".into(), 14, 18),
            ("35".into(), 19, 22),
            ("oclock".into(), 23, 31),
        ]
    );
    assert_eq!(
        texts("Hello, world! It's", KEEP_PUNCTUATION),
        ["Hello,", "world!", "It's"],
        "without the flag punctuation stays on its word"
    );
}

// Catches: an all-punctuation token kept, or counted.
#[test]
fn an_all_punctuation_token_is_dropped_unless_punctuation_is_kept() {
    assert_eq!(spans("hi ... there", DEFAULTS), [("hi".into(), 0, 2), ("there".into(), 7, 12)]);
    assert!(split_words("!!!", DEFAULTS).is_empty());
    assert_eq!(spans("!!!", KEEP_PUNCTUATION), [("!!!".into(), 0, 3)]);
}

// Catches: byte offsets, or a decomposed accent split from its letter.
#[test]
fn european_words_keep_accents_and_report_character_offsets() {
    assert_eq!(
        spans("C'est l'été, déjà !", DEFAULTS),
        [("Cest".into(), 0, 5), ("lété".into(), 6, 12), ("déjà".into(), 13, 17)]
    );
    assert_eq!(texts("¿Qué tal? ¡Muy bien!", DEFAULTS), ["Qué", "tal", "Muy", "bien"]);
    assert_eq!(
        spans("Größe: 5 Äpfel.", DEFAULTS),
        [("Größe".into(), 0, 6), ("5".into(), 7, 8), ("Äpfel".into(), 9, 15)]
    );
    assert_eq!(spans("e\u{301}te\u{301}", DEFAULTS), [("e\u{301}te\u{301}".into(), 0, 5)]);
}

// Catches: CJK left whole (no character split), or Latin inside it split per letter.
#[test]
fn chinese_is_one_word_per_character_and_latin_inside_it_stays_whole() {
    assert_eq!(
        spans("我用LiveKit开发。", KEEP_PUNCTUATION),
        [
            ("我".into(), 0, 1),
            ("用".into(), 1, 2),
            ("LiveKit".into(), 2, 9),
            ("开".into(), 9, 10),
            ("发。".into(), 10, 12),
        ]
    );
    assert_eq!(texts("第3章", DEFAULTS), ["第", "3", "章"]);
}

// Catches: hard-coded ranges that miss traditional forms, extension B or compatibility
// ideographs.
#[test]
fn every_han_block_splits_per_character() {
    assert_eq!(texts("這是繁體中文", DEFAULTS), ["這", "是", "繁", "體", "中", "文"]);
    assert_eq!(
        spans("𠮷野家", DEFAULTS),
        [("𠮷".into(), 0, 1), ("野".into(), 1, 2), ("家".into(), 2, 3)],
        "extension B, one character, not two"
    );
    assert_eq!(texts("\u{f900}\u{2f800}", DEFAULTS), ["\u{f900}", "\u{2f800}"]);
}

// Catches: punctuation split from CJK into a token of its own, or an opening bracket left
// alone instead of leading the next character.
#[test]
fn cjk_punctuation_stays_with_its_neighbour() {
    assert_eq!(
        spans("「はい」、人々。", KEEP_PUNCTUATION),
        [
            ("「は".into(), 0, 2),
            ("い」、".into(), 2, 5),
            ("人".into(), 5, 6),
            ("々。".into(), 6, 8)
        ]
    );
    assert_eq!(
        spans("你好，世界。", DEFAULTS),
        [("你".into(), 0, 1), ("好".into(), 1, 3), ("世".into(), 3, 4), ("界".into(), 4, 6)]
    );
}

// Catches: the prolonged sound mark or half-width katakana outside the set.
#[test]
fn japanese_kana_and_kanji_split_per_character() {
    assert_eq!(
        texts("東京タワーへ行きます", DEFAULTS),
        ["東", "京", "タ", "ワ", "ー", "へ", "行", "き", "ま", "す"]
    );
    assert_eq!(texts("ｺｰﾋｰ", DEFAULTS), ["ｺ", "ｰ", "ﾋ", "ｰ"]);
}

// Catches: a combining or half-width (han)dakuten, or a variation selector, made a word.
#[test]
fn marks_stay_on_their_base_character() {
    assert_eq!(
        spans("ｶﾞｲﾄﾞ", DEFAULTS),
        [("ｶﾞ".into(), 0, 2), ("ｲ".into(), 2, 3), ("ﾄﾞ".into(), 3, 5)]
    );
    assert_eq!(spans("か\u{3099}き", DEFAULTS), [("か\u{3099}".into(), 0, 2), ("き".into(), 2, 3)]);
    assert_eq!(texts("葛\u{e0100}城", DEFAULTS), ["葛\u{e0100}", "城"]);
}

// Catches: a virama, matra or nukta treated as a break.
#[test]
fn hindi_conjuncts_matras_and_nukta_stay_in_the_word() {
    assert_eq!(
        spans("क्षत्रिय नमस्ते, क़िला।", DEFAULTS),
        [("क्षत्रिय".into(), 0, 8), ("नमस्ते".into(), 9, 16), ("क़िला".into(), 17, 23)]
    );
    assert_eq!(
        texts("Kal meeting hai, theek है?", DEFAULTS),
        ["Kal", "meeting", "hai", "theek", "है"]
    );
}

// Catches: Hangul added to the per-character set.
#[test]
fn korean_splits_at_spaces() {
    assert_eq!(
        spans("안녕하세요, 반갑습니다!", DEFAULTS),
        [("안녕하세요".into(), 0, 6), ("반갑습니다".into(), 7, 13)]
    );
}

// Catches: per-character splitting of scripts without spaces between words; it separated
// Thai vowel and tone marks and broke TTS alignment (livekit/agents#3872).
#[test]
fn thai_lao_khmer_and_myanmar_split_only_at_spaces() {
    assert_eq!(
        spans("สวัสดีครับ ยินดีต้อนรับ", DEFAULTS),
        [("สวัสดีครับ".into(), 0, 10), ("ยินดีต้อนรับ".into(), 11, 23)]
    );
    for text in ["ສະບາຍດີ ເຈົ້າ", "សួស្តី ពិភពលោក", "မင်္ဂလာပါ ခင်ဗျာ"]
    {
        assert_eq!(texts(text, DEFAULTS), text.split(' ').collect::<Vec<_>>(), "{text}");
    }
}

// Catches: a skin tone or ZWJ sequence broken up, or an emoji glued to the CJK beside it.
#[test]
fn emoji_sequences_are_one_word() {
    let family = "👨\u{200d}👩\u{200d}👧";
    assert_eq!(texts(&format!("👍🏽 ok {family}"), DEFAULTS), ["👍🏽", "ok", family]);
    assert_eq!(texts(&format!("好{family}了"), DEFAULTS), ["好", family, "了"]);
    assert_eq!(texts("好👍🏽", DEFAULTS), ["好", "👍🏽"]);
}

// Catches: dropping the whitespace run under `retain_format`, attaching it to the previous
// token, or a whitespace-only token before CJK (`synchronizer.py:502` concatenates the tokens).
#[test]
fn retain_format_glues_whitespace_to_the_next_token() {
    assert_eq!(
        spans("a  b ", RETAIN),
        [("a".into(), 0, 1), ("  b".into(), 1, 4), (" ".into(), 4, 5)]
    );
    assert_eq!(
        spans("a 世界", RETAIN),
        [("a".into(), 0, 1), (" 世".into(), 1, 3), ("界".into(), 3, 4)]
    );
    for text in SAMPLES {
        for opts in [RETAIN, WordTokenizerOptions { ignore_punctuation: true, ..RETAIN }] {
            let words = split_words(text, opts);
            let joined: String = words.iter().map(|w| w.text.as_ref()).collect();
            let stripped: String =
                text.chars().filter(|c| !opts.ignore_punctuation || !is_punctuation(*c)).collect();
            assert_eq!(joined, stripped, "{text:?} with {opts:?}");
        }
    }
}

// Catches: offsets that overlap, run backwards, leave the text, or disagree with the token.
#[test]
fn every_span_is_its_tokens_source_in_order() {
    for text in SAMPLES {
        let chars: Vec<char> = text.chars().collect();
        for flags in 0..4u8 {
            let opts = WordTokenizerOptions {
                ignore_punctuation: flags & 1 != 0,
                retain_format: flags & 2 != 0,
            };
            let mut prev_end = 0;
            for w in split_words(text, opts) {
                assert!(prev_end <= w.start && w.start < w.end && w.end <= chars.len());
                let source: String = chars[w.start..w.end]
                    .iter()
                    .filter(|c| !opts.ignore_punctuation || !is_punctuation(**c))
                    .collect();
                assert_eq!(w.text, source, "{text:?} with {opts:?}");
                prev_end = w.end;
            }
        }
    }
}

// Catches: using `char::is_whitespace`.
#[test]
fn python_isspace_covers_the_c0_separators() {
    assert!(!'\u{1e}'.is_whitespace(), "the premise: Rust does not call this whitespace");
    assert_eq!(spans("a\u{1e}b", DEFAULTS), [("a".into(), 0, 1), ("b".into(), 2, 3)]);
    assert_eq!(spans("a\u{1e}b", RETAIN), [("a".into(), 0, 1), ("\u{1e}b".into(), 1, 3)]);
}

// Catches: an empty token for surrounding whitespace, or empty input treated as an error.
#[test]
fn empty_and_whitespace_only_input_produce_no_words() {
    for opts in [DEFAULTS, RETAIN] {
        assert!(split_words("", opts).is_empty());
    }
    assert!(split_words("   ", DEFAULTS).is_empty());
    assert_eq!(spans("   ", RETAIN), [("   ".into(), 0, 3)], "the source, reassembled");
    assert_eq!(spans(" hi ", DEFAULTS), [("hi".into(), 1, 3)]);
}

// Catches: non-ASCII punctuation kept on the word, or a symbol counted as punctuation.
#[test]
fn unicode_punctuation_is_punctuation() {
    assert!(is_punctuation('،') && is_punctuation('。') && is_punctuation('।'));
    assert!(!is_punctuation('€') && !is_punctuation('a'));
}

/// A CJK word stream releases each character once the next one starts, never holding a clause
/// until the end of input. https://github.com/livekit/agents/issues/1985
#[test]
fn cjk_word_streams_release_each_character_once_the_next_starts() {
    let opts = WordTokenizerOptions { ignore_punctuation: false, retain_format: true };
    for (text, pushed, flushed) in [
        ("你好世界。接下来", &["你", "好", "世", "界。", "接", "下"][..], "来"),
        ("こんにちは。次", &["こ", "ん", "に", "ち", "は。"][..], "次"),
    ] {
        let mut stream = WordStream::new(opts);
        let out: Vec<String> = text.chars().flat_map(|c| stream.push(&c.to_string())).collect();
        assert_eq!(out, pushed);
        assert_eq!(stream.flush(), [flushed]);
    }
}

// Catches: a word emitted before another follows it, so a word still arriving is cut.
#[test]
fn a_streamed_word_goes_out_once_another_follows_it() {
    let mut stream = WordStream::new(DEFAULTS);
    assert_eq!(stream.push("Hel"), Vec::<String>::new());
    assert_eq!(stream.push("lo, "), Vec::<String>::new(), "a space alone proves nothing");
    assert_eq!(stream.push("wo"), ["Hello"]);
    assert_eq!(stream.flush(), ["wo"]);
    assert_eq!(stream.flush(), Vec::<String>::new(), "flush is not repeatable");

    // Under `retain_format` the whitespace is a token of its own, so it proves the word whole.
    let mut stream = WordStream::new(RETAIN);
    assert_eq!(stream.push("Hello "), ["Hello"]);
    assert_eq!(stream.push("there"), Vec::<String>::new());
    assert_eq!(stream.flush(), [" there"]);
}

// Catches: a stream whose words depend on how the text was chunked.
#[test]
fn a_word_stream_says_what_split_words_says() {
    for text in SAMPLES {
        for flags in 0..4u8 {
            let opts = WordTokenizerOptions {
                ignore_punctuation: flags & 1 != 0,
                retain_format: flags & 2 != 0,
            };
            let mut stream = WordStream::new(opts);
            let mut out: Vec<String> =
                text.chars().flat_map(|c| stream.push(&c.to_string())).collect();
            out.extend(stream.flush());
            assert_eq!(out, texts(text, opts), "{text:?} with {opts:?}");
        }
    }
}
