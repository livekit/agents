//! Verbatim from `_language_data.py`. Sorted slices for `binary_search_by_key` (`tables_are_sorted`
//! guards it).

static ISO_639_3_TO_1: &[(&str, Option<&str>)] = &[
    ("afr", Some("af")),
    ("amh", Some("am")),
    ("ara", Some("ar")),
    ("asm", Some("as")),
    ("ast", None),
    ("aze", Some("az")),
    ("bel", Some("be")),
    ("ben", Some("bn")),
    ("bos", Some("bs")),
    ("bul", Some("bg")),
    ("cat", Some("ca")),
    ("ceb", None),
    ("ces", Some("cs")),
    ("cmn", Some("zh")),
    ("cym", Some("cy")),
    ("dan", Some("da")),
    ("deu", Some("de")),
    ("ell", Some("el")),
    ("eng", Some("en")),
    ("est", Some("et")),
    ("fas", Some("fa")),
    ("fil", None),
    ("fin", Some("fi")),
    ("fra", Some("fr")),
    ("ful", Some("ff")),
    ("gle", Some("ga")),
    ("glg", Some("gl")),
    ("guj", Some("gu")),
    ("hau", Some("ha")),
    ("heb", Some("he")),
    ("hin", Some("hi")),
    ("hrv", Some("hr")),
    ("hun", Some("hu")),
    ("hye", Some("hy")),
    ("ibo", Some("ig")),
    ("ind", Some("id")),
    ("isl", Some("is")),
    ("ita", Some("it")),
    ("jav", Some("jv")),
    ("jpn", Some("ja")),
    ("kan", Some("kn")),
    ("kat", Some("ka")),
    ("kaz", Some("kk")),
    ("kea", None),
    ("khm", Some("km")),
    ("kir", Some("ky")),
    ("kor", Some("ko")),
    ("kur", Some("ku")),
    ("lao", Some("lo")),
    ("lav", Some("lv")),
    ("lin", Some("ln")),
    ("lit", Some("lt")),
    ("ltz", Some("lb")),
    ("lug", Some("lg")),
    ("luo", None),
    ("mal", Some("ml")),
    ("mar", Some("mr")),
    ("mkd", Some("mk")),
    ("mlt", Some("mt")),
    ("mon", Some("mn")),
    ("mri", Some("mi")),
    ("msa", Some("ms")),
    ("mya", Some("my")),
    ("nep", Some("ne")),
    ("nld", Some("nl")),
    ("nor", Some("no")),
    ("nso", None),
    ("nya", Some("ny")),
    ("oci", Some("oc")),
    ("ori", Some("or")),
    ("pan", Some("pa")),
    ("pol", Some("pl")),
    ("por", Some("pt")),
    ("pus", Some("ps")),
    ("ron", Some("ro")),
    ("rus", Some("ru")),
    ("slk", Some("sk")),
    ("slv", Some("sl")),
    ("sna", Some("sn")),
    ("snd", Some("sd")),
    ("som", Some("so")),
    ("spa", Some("es")),
    ("srp", Some("sr")),
    ("swa", Some("sw")),
    ("swe", Some("sv")),
    ("tam", Some("ta")),
    ("tel", Some("te")),
    ("tgk", Some("tg")),
    ("tha", Some("th")),
    ("tur", Some("tr")),
    ("ukr", Some("uk")),
    ("umb", None),
    ("urd", Some("ur")),
    ("uzb", Some("uz")),
    ("vie", Some("vi")),
    ("wol", Some("wo")),
    ("xho", Some("xh")),
    ("yue", None),
    ("zho", Some("zh")),
    ("zul", Some("zu")),
];

static NAME_TO_CODE: &[(&str, &str)] = &[
    ("afrikaans", "af"),
    ("albanian", "sq"),
    ("amharic", "am"),
    ("arabic", "ar"),
    ("armenian", "hy"),
    ("azerbaijani", "az"),
    ("basque", "eu"),
    ("belarusian", "be"),
    ("bengali", "bn"),
    ("bosnian", "bs"),
    ("bulgarian", "bg"),
    ("burmese", "my"),
    ("catalan", "ca"),
    ("chinese", "zh"),
    ("croatian", "hr"),
    ("czech", "cs"),
    ("danish", "da"),
    ("dutch", "nl"),
    ("english", "en"),
    ("estonian", "et"),
    ("finnish", "fi"),
    ("french", "fr"),
    ("galician", "gl"),
    ("georgian", "ka"),
    ("german", "de"),
    ("greek", "el"),
    ("gujarati", "gu"),
    ("hausa", "ha"),
    ("hebrew", "he"),
    ("hindi", "hi"),
    ("hungarian", "hu"),
    ("icelandic", "is"),
    ("indonesian", "id"),
    ("irish", "ga"),
    ("italian", "it"),
    ("japanese", "ja"),
    ("javanese", "jv"),
    ("kannada", "kn"),
    ("kazakh", "kk"),
    ("khmer", "km"),
    ("korean", "ko"),
    ("kurdish", "ku"),
    ("kyrgyz", "ky"),
    ("lao", "lo"),
    ("latvian", "lv"),
    ("lingala", "ln"),
    ("lithuanian", "lt"),
    ("luxembourgish", "lb"),
    ("macedonian", "mk"),
    ("malay", "ms"),
    ("malayalam", "ml"),
    ("maltese", "mt"),
    ("maori", "mi"),
    ("marathi", "mr"),
    ("mongolian", "mn"),
    ("nepali", "ne"),
    ("norwegian", "no"),
    ("occitan", "oc"),
    ("oriya", "or"),
    ("pashto", "ps"),
    ("persian", "fa"),
    ("polish", "pl"),
    ("portuguese", "pt"),
    ("punjabi", "pa"),
    ("romanian", "ro"),
    ("russian", "ru"),
    ("serbian", "sr"),
    ("shona", "sn"),
    ("sindhi", "sd"),
    ("slovak", "sk"),
    ("slovene", "sl"),
    ("slovenian", "sl"),
    ("somali", "so"),
    ("spanish", "es"),
    ("swahili", "sw"),
    ("swedish", "sv"),
    ("tagalog", "tl"),
    ("tajik", "tg"),
    ("tamil", "ta"),
    ("telugu", "te"),
    ("thai", "th"),
    ("turkish", "tr"),
    ("ukrainian", "uk"),
    ("urdu", "ur"),
    ("uzbek", "uz"),
    ("vietnamese", "vi"),
    ("welsh", "cy"),
    ("wolof", "wo"),
    ("xhosa", "xh"),
    ("yoruba", "yo"),
    ("zulu", "zu"),
];

/// The reverse of [`NAME_TO_CODE`]. Two names map to `sl`, and it must be `"slovene"` (punkt only
/// ships that one).
static CODE_TO_NAME: std::sync::LazyLock<std::collections::HashMap<&str, &str>> =
    std::sync::LazyLock::new(|| {
        NAME_TO_CODE.iter().filter(|(name, _)| *name != "slovenian").map(|&(n, c)| (c, n)).collect()
    });

fn get<V: Copy>(table: &[(&'static str, V)], key: &str) -> Option<V> {
    table.binary_search_by_key(&key, |(k, _)| k).ok().map(|i| table[i].1)
}

/// `Some(None)` for codes with no 639-1 equivalent: provenance rows that behave like absent ones and
/// stop someone "adding" `yue -> zh`.
pub(crate) fn iso_639_3_to_1(code: &str) -> Option<Option<&'static str>> {
    get(ISO_639_3_TO_1, code)
}

/// The NLTK punkt set plus common names.
pub(crate) fn name_to_code(name: &str) -> Option<&'static str> {
    get(NAME_TO_CODE, name)
}

pub(crate) fn code_to_name(code: &str) -> Option<&'static str> {
    CODE_TO_NAME.get(code).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: an out-of-order row, which silently breaks the binary search.
    #[test]
    fn tables_are_sorted() {
        assert!(ISO_639_3_TO_1.windows(2).all(|w| w[0].0 < w[1].0), "ISO_639_3_TO_1");
        assert!(NAME_TO_CODE.windows(2).all(|w| w[0].0 < w[1].0), "NAME_TO_CODE");
    }

    // Catches: dropping the `sl` tie-break (punkt only ships "slovene").
    #[test]
    fn slovene_wins_the_duplicate_name_tiebreak() {
        assert_eq!(name_to_code("slovene"), Some("sl"));
        assert_eq!(name_to_code("slovenian"), Some("sl"));
        assert_eq!(code_to_name("sl"), Some("slovene"));
    }

    // Catches: a table edit that makes the name-vs-639-3 lookup order observable.
    #[test]
    fn the_two_input_tables_never_disagree() {
        for (name, code) in NAME_TO_CODE {
            if let Some(mapped) = iso_639_3_to_1(name) {
                assert_eq!(mapped, Some(*code), "{name} resolves differently in the two tables");
            }
        }
    }
}
