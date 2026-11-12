//! Where each word of a reply starts in its audio, for a voice that sends no word timings.
//!
//! A voice that times its words (Cartesia, Inworld, ElevenLabs) never comes here: its own timings
//! are used. For any other voice the transcript still has to show each word when it is said, and
//! all we know is the reply's text and its audio. The audio tells us one thing reliably: where it
//! is silent. This module turns text + silences into a start time per word.
//!
//! # The idea
//!
//! Voices pause at punctuation. So the silences in the audio and the breaks in the text (`.`, `,`,
//! a list item) are two sequences that should line up:
//!
//! ```text
//! text:   Thanks for calling.   Your order shipped,   it arrives Friday.
//!                            ^                      ^
//! audio:  ~~~~~~~~~~~~~~~~~ [0.5 s]  ~~~~~~~~~~~~~~ [0.3 s]  ~~~~~~~~~~~~~~
//! ```
//!
//! Once a silence is paired with a break, the words after that break start where the silence ends.
//! Between two pairs (a stretch), the stretch's speech is shared by its words by how long each
//! takes to say. So there are three steps:
//!
//! 1. **Weigh each word** ([`weight`]): how long it takes to say relative to the others. A small
//!    linear model of the word's log speaking time, from its letters, digits, syllables as read
//!    aloud (a number or an order ID is read out, `#W2166301` takes far longer than its length
//!    says), its script (CJK, Hangul, Devanagari), whether it is a short function word and the
//!    punctuation after it. Fitted on the word timings of 33 voices; no voice, provider or language
//!    is an input. The weights only matter relative to each other: the reply's own speaking rate
//!    (weight per second) is learnt from its audio.
//! 2. **Find each word's break** ([`breaks`]): a sentence end, a clause break (comma, colon, dash),
//!    or a bare line break; punctuation inside a closing quote or bracket counts (`"yes."`).
//! 3. **Pair silences with breaks** ([`Problem::pairs`]): a dynamic program over every way to pair them in
//!    order, keeping the cheapest. The cost of a pairing says how unlikely it is:
//!    - a stretch whose speech is much longer or shorter than its words' weights predict;
//!    - a break with no pause (cheap after a comma, expensive after a sentence end, which voices
//!      pause at 96% of the time), or a pause whose length is unusual for its break;
//!    - a silence matched to no break: a hesitation, which voices do about once per 4.6 s, usually
//!      short. A long silence in the middle of a phrase is unlikely to be one.
//!
//!    The constants ([`Break::pausing`], `HESITATION`, `SPREAD`, `FLOOR`, `FIT`) are measured on 918
//!    recorded replies of 40 voices in 11 languages.
//!
//! Then each stretch's words are spread over its speech by weight, skipping any unmatched silence
//! inside it ([`Problem::starts`]).
//!
//! # While the audio is still arriving
//!
//! The pacer calls [`place`] again whenever text, a pause or the end arrives. Until the audio is
//! complete its end is unknown ([`End::Open`]): the speaking rate starts at [`PRIOR_RATE`] and is
//! re-learnt from the stretches matched so far, and the last stretch is only penalized if more
//! audio has arrived than its words should need. Once the audio is complete ([`End::Known`]) the
//! reply's own rate is exact and the whole alignment is redone.
//!
//! # How good it is, and where it fails
//!
//! On the recorded corpus a word shows about 0.1 s from when it is said (median), within 0.74 s
//! for 90% of words. It fails when one word takes far longer than predicted (an ID read digit by
//! digit, a US price read in German): the pauses inside that reading get paired with later breaks
//! and every word after it is off by seconds until the next correct pair. Even perfect word
//! lengths would leave about 1.5 s at the 99th percentile; going below needs something that hears
//! the words (the voice's own timings).
//!
//! `transcript_sync_holds_its_accuracy` replays the replies checked in under `sync_recordings/`;
//! `make bench-sync` measures the full corpus and logs its drift across commits.

/// A word with no timing of its own, as the transcript holds it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Word {
    pub text: String,
    /// A line or list break follows it.
    pub line_break: bool,
    /// Punctuation said as no word of its own follows it (a lone dash).
    pub mark: Option<Break>,
    /// The reply's last word.
    pub last: bool,
}

impl Word {
    pub fn new(text: impl Into<String>) -> Word {
        Word { text: text.into(), line_break: false, mark: None, last: false }
    }

    /// The token after it: a line break before that token, or punctuation that is no word (a lone
    /// dash), is a break after this word.
    pub fn followed_by(&mut self, token: &str) {
        let lead = &token[..token.len() - token.trim_start().len()];
        self.line_break |= lead.contains('\n');
        if weight(token, false, false) == 0.0 {
            self.mark = self.mark.or(breaks(token, false));
        }
    }

    fn weighed(&self) -> Weighed {
        Weighed {
            weight: weight(&self.text, self.line_break, self.last),
            breaks: breaks(&self.text, self.line_break).or(self.mark),
        }
    }
}

/// A word as the alignment reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Weighed {
    /// Its relative speaking time ([`weight`]).
    weight: f64,
    /// The break after it, where a voice may pause.
    breaks: Option<Break>,
}

/// A silence, in seconds of the reply's audio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Span {
    pub start: f64,
    pub end: f64,
}

impl Span {
    fn len(&self) -> f64 {
        self.end - self.start
    }
}

/// Where the speech ends: known once the audio is complete, else only the audio received so far.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum End {
    Known(f64),
    Open { received: f64 },
}

/// Speaking rate, in weight per second of speech, before a reply's own is known: the median over
/// the 33 voices the weights were fitted on (interquartile range 5.2 to 6.3).
pub(crate) const PRIOR_RATE: f64 = 5.74;

/// Each word's start, in seconds of audio. `pauses` are the silences inside the speech, `start`
/// is where the speech begins; `speed` scales the prior rate while the end is still open.
pub(crate) fn place(words: &[Word], pauses: &[Span], start: f64, end: End, speed: f64) -> Vec<f64> {
    let weighed: Vec<Weighed> = words.iter().map(Word::weighed).collect();
    place_weighed(&weighed, pauses, start, end, speed)
}

fn place_weighed(words: &[Weighed], pauses: &[Span], start: f64, end: End, speed: f64) -> Vec<f64> {
    let problem = Problem::new(words, pauses, start);
    match end {
        End::Known(end) => {
            // the audio is complete: the reply's own rate is its weight over its speech
            let speech = end - start - pauses.iter().map(Span::len).sum::<f64>();
            let total = problem.cum[words.len()];
            let rate =
                if speech > 0.0 && total > 0.0 { total / speech } else { PRIOR_RATE * speed };
            let pairs = problem.pairs(End::Known(end), rate);
            problem.starts(&pairs, End::Known(end), rate)
        }
        End::Open { received } => {
            // align at the prior rate, learn the rate from what matched, align again
            let end = End::Open { received };
            let mut rate = PRIOR_RATE * speed;
            let mut pairs = problem.pairs(end, rate);
            rate = problem.learnt_rate(&pairs).unwrap_or(rate);
            pairs = problem.pairs(end, rate);
            problem.starts(&pairs, end, rate)
        }
    }
}

// Step 1: how long each word takes to say, relative to the others.

/// The model's weights, in the order of [`features`].
const COEFFICIENTS: [f64; 15] = [
    -1.163_340_169_712_863_8,  // bias
    0.655_155_935_203_209_4,   // ln(1 + letters)
    1.058_828_853_012_231_4,   // ln(1 + digits)
    -0.149_030_887_075_118_22, // ln(1 + capitals)
    0.646_573_277_177_422_1,   // an acronym
    -0.078_103_456_110_104_43, // a code or identifier
    0.426_576_838_480_590_97,  // ln(1 + CJK characters)
    0.348_429_800_814_110_2,   // ln(1 + Hangul syllables)
    0.120_371_304_731_121_4,   // ln(1 + Devanagari characters)
    -0.399_355_108_435_887,    // a function word
    0.153_792_325_698_402_54,  // a comma, colon or semicolon after it
    0.067_931_514_612_175_03,  // a sentence end after it
    0.187_360_264_154_405_82,  // a line or list break after it
    -0.033_355_923_232_015_54, // the reply's last word
    0.500_794_638_919_559_4,   // ln(1 + syllables as read aloud)
];

/// Stripped from a word's ends before its letters are counted.
const STRIP: &[char] = &[
    '.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '"', '\'', '“', '”', '‘', '’', '*', '_', '`',
    '#', '-', '•',
];
/// The punctuation the model reads at a word's end, as it was fitted (the breaks are [`breaks`]'s).
const TAIL: &[char] = &['.', ',', ';', ':', '!', '?', '。', '！', '？', '，', '、', '।'];
/// Short words read quickly, in the corpus's languages.
const FUNCTION: &[&str] = &[
    "the", "a", "an", "to", "of", "and", "or", "in", "on", "at", "for", "is", "it", "be", "you",
    "your", "we", "i", "my", "me", "our", "this", "that", "with", "as", "are", "was", "by", "from",
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "à", "au", "est", "pour", "que",
    "qui", "il", "elle", "vous", "nous", "ne", "pas", "el", "los", "las", "del", "y", "una", "por",
    "con", "para", "es", "se", "no", "lo", "su", "o", "os", "do", "da", "dos", "das", "um", "uma",
    "e", "em", "na", "com",
];

/// The word's relative speaking time; 0 for a word with nothing to say (punctuation alone).
pub(crate) fn weight(word: &str, line_break: bool, last: bool) -> f64 {
    if !word.chars().any(said) {
        return 0.0;
    }
    features(word, line_break, last).iter().zip(COEFFICIENTS).map(|(f, c)| f * c).sum::<f64>().exp()
}

/// What the model reads of a word, in the order of [`COEFFICIENTS`].
///
/// The weight is `exp(Σ feature × coefficient)`. So a flag (0 or 1) multiplies the weight by
/// `exp(coefficient)` when set, and a count `n` read as `ln(1 + n)` multiplies it by
/// `(1 + n)^coefficient`: the weight grows with the count, but less than in proportion. Counts are
/// taken on the word's core, its surrounding punctuation stripped ([`STRIP`]).
///
/// | # | Feature | Why it matters | Effect |
/// |---|---|---|---|
/// | 0 | bias | the scale of every weight | `exp(-1.16)` |
/// | 1 | `ln(1 + letters)` | longer words take longer | `(1 + n)^0.66` |
/// | 2 | `ln(1 + digits)` | each digit is a spoken word or more | `(1 + n)^1.06` |
/// | 3 | `ln(1 + capitals)` | capitals the syllables already count (a name, a sentence start) | `(1 + n)^-0.15` |
/// | 4 | an acronym (2+ capitals, all of its letters) | spelt letter by letter: "SMS" | ×1.91 |
/// | 5 | a code (a digit beside a letter, or `_/@#`) | its syllables already count it | ×0.92 |
/// | 6 | `ln(1 + CJK characters)` | a kana or a hanzi is about one syllable each | `(1 + n)^0.43` |
/// | 7 | `ln(1 + Hangul blocks)` | one syllable each | `(1 + n)^0.35` |
/// | 8 | `ln(1 + Devanagari characters)` | on top of its vowel count | `(1 + n)^0.12` |
/// | 9 | a function word ("the", "de", "y") | said quickly, unstressed | ×0.67 |
/// | 10 | a comma, colon or semicolon after it | voices slow into a pause | ×1.17 |
/// | 11 | a sentence end after it | the same, less (the pause itself is not in the weight) | ×1.07 |
/// | 12 | a break after it (as fitted: any punctuation break or a line break) | the same, for a list item | ×1.21 |
/// | 13 | the reply's last word | said no longer than the others | ×0.97 |
/// | 14 | `ln(1 + syllables as read aloud)` | the strongest cue: a number or an ID read out ([`syllables`]) | `(1 + n)^0.50` |
///
/// Features 10 and 11 read the word's own trailing punctuation ([`TAIL`]), as the model was
/// fitted; where a voice may pause is decided separately, by [`breaks`].
fn features(word: &str, line_break: bool, last: bool) -> [f64; 15] {
    let w = word.trim();
    let core = w.trim_matches(STRIP);
    let (mut letters, mut digits, mut upper, mut alpha) = (0u32, 0u32, 0u32, 0u32);
    let (mut cjk, mut hangul, mut deva) = (0u32, 0u32, 0u32);
    for c in core.chars() {
        if is_cjk(c) {
            cjk += 1;
        } else if is_hangul(c) {
            hangul += 1;
        } else if ('\u{900}'..='\u{97f}').contains(&c) {
            deva += 1;
        }
        letters += u32::from(c.is_alphabetic() || is_mark(c));
        digits += u32::from(c.is_numeric());
        upper += u32::from(c.is_uppercase());
        alpha += u32::from(c.is_alphabetic() && !is_mark(c));
    }
    let tail = w.trim_end_matches(TAIL);
    let ends = w[tail.len()..].chars().last();
    let ln = |n: u32| f64::from(n).ln_1p();
    let flag = |b: bool| f64::from(u8::from(b));
    [
        1.0,
        ln(letters),
        ln(digits),
        ln(upper),
        flag(upper >= 2 && upper == alpha),
        flag(is_code(core)),
        ln(cjk),
        ln(hangul),
        ln(deva),
        flag(FUNCTION.contains(&core.to_lowercase().as_str())),
        flag(ends.is_some_and(|c| ",，、;:".contains(c))),
        flag(ends.is_some_and(|c| ".!?。！？।".contains(c))),
        flag(line_break),
        flag(last),
        syllables(word).ln_1p(),
    ]
}

/// Syllables as a voice reads the word: a CJK character, a Hangul block, a Devanagari vowel each;
/// a number as spoken; a code character by character; else its vowel groups.
fn syllables(word: &str) -> f64 {
    if word.chars().any(is_cjk) {
        return word.chars().filter(|&c| said(c)).count() as f64;
    }
    if word.chars().any(is_hangul) {
        return word.chars().filter(|&c| is_hangul(c)).count() as f64;
    }
    if word.chars().any(|c| ('\u{900}'..='\u{97f}').contains(&c)) {
        let vowels = word.chars().filter(|&c| is_devanagari_vowel(c)).count();
        return (vowels + 1).max(1) as f64;
    }
    if let Some(n) = numbers::spoken(word) {
        return n;
    }
    if is_code_like(word) {
        return code_syllables(word).max(1.0);
    }
    if word.chars().any(said) { vowel_groups(word).max(1) as f64 } else { 0.0 }
}

/// `[0-9_/@#]|[A-Z]{2,}|[a-z][A-Z]` anywhere: a code, an identifier or an acronym.
fn is_code_like(word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    chars.iter().any(|c| c.is_ascii_digit() || "_/@#".contains(*c))
        || chars.windows(2).any(|p| {
            (p[0].is_ascii_uppercase() && p[1].is_ascii_uppercase())
                || (p[0].is_ascii_lowercase() && p[1].is_ascii_uppercase())
        })
}

/// A code read piece by piece: digits and capitals one by one, a word by its vowel groups.
fn code_syllables(word: &str) -> f64 {
    let chars: Vec<char> = word.chars().collect();
    let mut n = 0.0;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let lower_run = chars[i + 1..].iter().take_while(|c| c.is_ascii_lowercase()).count();
        if c.is_ascii_alphabetic() && lower_run > 0 {
            let part: String = chars[i..=i + lower_run].iter().collect();
            n += f64::from(vowel_groups(&part).max(1));
            i += 1 + lower_run;
        } else if c.is_ascii_uppercase() {
            let mut run = chars[i..].iter().take_while(|c| c.is_ascii_uppercase()).count();
            if chars.get(i + run).is_some_and(char::is_ascii_lowercase) {
                run -= 1; // the last capital starts the next word
            }
            n += if run == 1 { 1.2 } else { 1.2 * run as f64 };
            i += run;
        } else if c.is_numeric() {
            n += 1.4;
            i += 1;
        } else if c.is_ascii_lowercase() || c.is_whitespace() {
            i += 1; // a lone lowercase letter is no piece
        } else {
            if c.is_alphabetic() {
                n += 1.2;
            } else if "/@#".contains(c) {
                n += 1.0;
            }
            i += 1;
        }
    }
    n
}

fn vowel_groups(word: &str) -> u32 {
    let mut n = 0;
    let mut inside = false;
    for c in word.chars().flat_map(char::to_lowercase) {
        let vowel = "aeiouyàâäéèêëîïôöùûüáíóúãõœæ".contains(c);
        n += u32::from(vowel && !inside);
        inside = vowel;
    }
    n
}

/// Something a voice says: a letter, a digit or a mark on one.
fn said(c: char) -> bool {
    c.is_alphanumeric() || is_mark(c)
}

fn is_cjk(c: char) -> bool {
    ('\u{3040}'..='\u{30ff}').contains(&c) || ('\u{3400}'..='\u{9fff}').contains(&c)
}

fn is_hangul(c: char) -> bool {
    ('\u{ac00}'..='\u{d7af}').contains(&c)
}

fn is_devanagari_vowel(c: char) -> bool {
    ('\u{905}'..='\u{914}').contains(&c) || ('\u{93e}'..='\u{94c}').contains(&c)
}

/// A combining mark in the scripts the model was fitted on.
fn is_mark(c: char) -> bool {
    matches!(c,
        '\u{300}'..='\u{36f}' | '\u{900}'..='\u{903}' | '\u{93a}'..='\u{94f}'
        | '\u{951}'..='\u{957}' | '\u{962}'..='\u{963}' | '\u{1ab0}'..='\u{1aff}'
        | '\u{1dc0}'..='\u{1dff}' | '\u{20d0}'..='\u{20ff}' | '\u{3099}'..='\u{309a}'
        | '\u{fe20}'..='\u{fe2f}')
}

/// `[0-9]` neighbouring an ASCII letter, or one of `_/@#`: a code or an identifier.
fn is_code(core: &str) -> bool {
    let chars: Vec<char> = core.chars().collect();
    chars.iter().any(|c| "_/@#".contains(*c))
        || chars.windows(2).any(|p| {
            (p[0].is_numeric() && p[1].is_ascii_alphabetic())
                || (p[0].is_ascii_alphabetic() && p[1].is_numeric())
        })
}

// Step 2: where a voice may pause, after which word, and how likely it is to.

/// A place in the text a voice may pause after.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Break {
    /// `.`, `!`, `?` and their kin: nearly always a pause, a long one.
    Sentence,
    /// `,`, `;`, `:`, a dash: often a pause, a short one.
    Clause,
    /// A line or list break with no punctuation.
    Line,
}

/// The punctuation that ends a sentence, where voices nearly always pause: Latin, CJK, Devanagari,
/// Arabic and Ethiopic marks and the ellipsis.
const SENTENCE: &[char] = &['.', '!', '?', '…', '。', '！', '？', '।', '॥', '؟', '።'];
/// The punctuation inside a sentence a voice often pauses at, Greek's question mark among them.
const CLAUSE: &[char] =
    &[',', ';', ':', '—', '–', '，', '、', '；', '：', '،', '؛', '\u{37e}', '፣', '፤'];
/// What may close a phrase after its punctuation: `"yes."`, `(see below).` and `«oui.»` break.
const CLOSERS: &[char] = &[
    '"', '\'', '’', '”', '»', '›', ')', ']', '}', '」', '』', '）', '】', '〉', '》', '*', '_', '`',
];

/// The break after `text`, if a voice may pause there: punctuation, maybe inside a closing quote
/// or bracket, else a line break after it.
fn breaks(text: &str, line_break: bool) -> Option<Break> {
    let text = text.trim_end().trim_end_matches(CLOSERS);
    if text.ends_with(SENTENCE) {
        Some(Break::Sentence)
    } else if text.ends_with(CLAUSE) {
        Some(Break::Clause)
    } else {
        line_break.then_some(Break::Line)
    }
}

/// How voices pause, measured on 918 replies of 40 voices in 11 languages: the chance of a pause,
/// and its length's log mean and spread, in seconds.
struct Pausing {
    chance: f64,
    log_mean: f64,
    log_sd: f64,
}

impl Break {
    fn pausing(self) -> Pausing {
        match self {
            Break::Sentence => Pausing { chance: 0.964, log_mean: -0.714, log_sd: 0.484 },
            Break::Clause => Pausing { chance: 0.717, log_mean: -1.015, log_sd: 0.405 },
            Break::Line => Pausing { chance: 0.739, log_mean: -0.576, log_sd: 0.711 },
        }
    }
}

/// A pause where the text has no break: a hesitation, about one per 4.6 s of speech, its chance
/// read per 150 ms of it.
const HESITATION: Pausing = Pausing { chance: 0.215 * 0.15, log_mean: -1.001, log_sd: 0.488 };
/// A stretch's speech varies around its words' expected length by `SPREAD * expected + FLOOR`
/// seconds squared; `FIT` weighs that against the pauses.
const SPREAD: f64 = 0.25;
const FLOOR: f64 = 0.25;
const FIT: f64 = 6.0;

impl Pausing {
    /// The cost of a pause of `secs` here: its negative log-likelihood.
    fn paused(&self, secs: f64) -> f64 {
        let secs = secs.max(1e-3);
        let z = (secs.ln() - self.log_mean) / self.log_sd;
        -self.chance.ln() + 0.5 * z * z + (self.log_sd * secs).ln()
    }

    /// The cost of no pause here.
    fn passed(&self) -> f64 {
        -(1.0 - self.chance).ln()
    }
}

/// The cost of `speech` seconds where the words expect `expected`.
fn fit(speech: f64, expected: f64) -> f64 {
    let var = SPREAD * expected + FLOOR;
    FIT * (0.5 * (speech - expected).powi(2) / var + 0.5 * var.ln())
}

// Step 3: pairing the silences with the breaks.

/// Breaks or pauses one stretch may span.
const WINDOW: usize = 14;

/// One reply's alignment. Breaks and pauses are numbered from 1 here, 0 meaning "before the
/// first": a stretch runs from just after break `k0` and pause `m0` to break `k1` and pause `m1`,
/// and an alignment is the sequence of `(break, pause)` pairs it matched.
struct Problem<'a> {
    pauses: &'a [Span],
    start: f64,
    /// Each break: the word it follows, and its kind.
    breaks: Vec<(usize, Break)>,
    /// The weight before each word, and the total last.
    cum: Vec<f64>,
    /// Running sums, so a stretch's sums are two lookups: the pauses' lengths, their cost as
    /// hesitations, and the breaks' cost of passing with no pause.
    silence: Vec<f64>,
    hesitating: Vec<f64>,
    passing: Vec<f64>,
}

/// `(break, pause)`, numbered from 1.
type Node = (usize, usize);

impl<'a> Problem<'a> {
    fn new(words: &[Weighed], pauses: &'a [Span], start: f64) -> Problem<'a> {
        let last = words.len().saturating_sub(1);
        let breaks: Vec<(usize, Break)> =
            (0..last).filter_map(|i| words[i].breaks.map(|b| (i, b))).collect();
        Problem {
            pauses,
            start,
            cum: running(words.iter().map(|w| w.weight)),
            silence: running(pauses.iter().map(Span::len)),
            hesitating: running(pauses.iter().map(|p| HESITATION.paused(p.len()))),
            passing: running(breaks.iter().map(|(_, b)| b.pausing().passed())),
            breaks,
        }
    }

    /// The weight said before break `k`'s next word; past the last break, the reply's.
    fn weight_at(&self, k: usize) -> f64 {
        match k {
            0 => 0.0,
            k if k > self.breaks.len() => self.cum[self.cum.len() - 1],
            k => self.cum[self.first_word(k)],
        }
    }

    /// The first word after break `k`.
    fn first_word(&self, k: usize) -> usize {
        if k == 0 {
            0
        } else if k > self.breaks.len() {
            self.cum.len() - 1
        } else {
            self.breaks[k - 1].0 + 1
        }
    }

    /// Where speech resumes after pause `m`; after none, where it starts.
    fn resumes(&self, m: usize) -> f64 {
        if m == 0 { self.start } else { self.pauses[m - 1].end }
    }

    /// The pauses strictly between pauses `m0` and `m1`.
    fn inner(&self, m0: usize, m1: usize) -> &'a [Span] {
        let m_end = self.pauses.len();
        &self.pauses[m0.min(m_end)..m1.saturating_sub(1).min(m_end)]
    }

    /// The cost of the stretch from `(k0, m0)` to break `k1` matched with pause `m1`; `None` if
    /// it has no speech.
    fn stretch(&self, (k0, m0): Node, (k1, m1): Node, rate: f64) -> Option<f64> {
        let paused = self.pauses[m1 - 1];
        let speech = paused.start - self.resumes(m0) - between(&self.silence, m0, m1);
        (speech > 0.0).then(|| {
            fit(speech, (self.weight_at(k1) - self.weight_at(k0)) / rate)
                + self.breaks[k1 - 1].1.pausing().paused(paused.len())
                + between(&self.passing, k0, k1)
                + between(&self.hesitating, m0, m1)
        })
    }

    /// The cost of the last stretch, from `(k0, m0)` to the end of the reply; `None` if it has
    /// no speech.
    fn last(&self, (k0, m0): Node, end: End, rate: f64) -> Option<f64> {
        let (k_end, m_end) = (self.breaks.len(), self.pauses.len());
        let silent = self.silence[m_end] - self.silence[m0];
        let expected = (self.weight_at(k_end + 1) - self.weight_at(k0)) / rate;
        let cost = match end {
            End::Known(end) => {
                let speech = end - self.resumes(m0) - silent;
                if speech <= 0.0 {
                    return None;
                }
                fit(speech, expected) + between(&self.passing, k0, k_end + 1)
            }
            End::Open { received } => {
                // only the speech received is known: penalized only past what the words
                // expect, and only the breaks it has reached were passed with no pause
                let spoken = received - self.resumes(m0) - silent;
                let reached = self.breaks[k0.min(k_end)..]
                    .iter()
                    .take_while(|(i, _)| (self.cum[i + 1] - self.weight_at(k0)) / rate < spoken)
                    .count();
                fit(spoken.max(expected), expected) - fit(expected, expected)
                    + between(&self.passing, k0, k0 + reached + 1)
            }
        };
        Some(cost + self.hesitating[m_end] - self.hesitating[m0])
    }

    /// The cheapest alignment, as `(break, pause)` pairs numbered from 0.
    fn pairs(&self, end: End, rate: f64) -> Vec<(usize, usize)> {
        let (k_end, m_end) = (self.breaks.len(), self.pauses.len());
        // best[k][m]: the cheapest alignment whose last pair is (k, m), and the pair before it
        let mut best = vec![vec![None::<(f64, Node)>; m_end + 1]; k_end + 1];
        best[0][0] = Some((0.0, (0, 0)));
        for k1 in 1..=k_end {
            for m1 in 1..=m_end {
                let mut cheapest: Option<(f64, Node)> = None;
                for k0 in k1.saturating_sub(WINDOW)..k1 {
                    for m0 in m1.saturating_sub(WINDOW)..m1 {
                        let Some((before, _)) = best[k0][m0] else { continue };
                        let Some(c) = self.stretch((k0, m0), (k1, m1), rate) else { continue };
                        if cheapest.is_none_or(|(best, _)| before + c < best) {
                            cheapest = Some((before + c, (k0, m0)));
                        }
                    }
                }
                best[k1][m1] = cheapest;
            }
        }
        let mut last: Option<(f64, Node)> = None;
        for (k0, row) in best.iter().enumerate() {
            for (m0, cell) in row.iter().enumerate() {
                let Some((before, _)) = cell else { continue };
                let Some(c) = self.last((k0, m0), end, rate) else { continue };
                if last.is_none_or(|(best, _)| before + c < best) {
                    last = Some((before + c, (k0, m0)));
                }
            }
        }
        let mut pairs = Vec::new();
        let mut node = last.map_or((0, 0), |(_, at)| at);
        while node != (0, 0) {
            pairs.push((node.0 - 1, node.1 - 1));
            node = best[node.0][node.1].map_or((0, 0), |(_, from)| from);
        }
        pairs.reverse();
        pairs
    }

    /// Each word's start: every stretch's speech spread over its words by weight, the unmatched
    /// pauses inside it skipped.
    fn starts(&self, pairs: &[(usize, usize)], end: End, rate: f64) -> Vec<f64> {
        let (k_end, m_end) = (self.breaks.len(), self.pauses.len());
        let mut starts = vec![self.start; self.cum.len() - 1];
        let mut bounds: Vec<Node> = vec![(0, 0)];
        bounds.extend(pairs.iter().map(|&(k, m)| (k + 1, m + 1)));
        bounds.push((k_end + 1, m_end + 1));
        for pair in bounds.windows(2) {
            let ((k0, m0), (k1, m1)) = (pair[0], pair[1]);
            let (i0, i1) = (self.first_word(k0), self.first_word(k1));
            let resumes = self.resumes(m0);
            let inner = self.inner(m0, m1);
            let silent: f64 = inner.iter().map(Span::len).sum();
            let speech = match (k1 > k_end, end) {
                (false, _) => Some(self.pauses[m1 - 1].start - resumes - silent),
                (true, End::Known(end)) => Some(end - resumes - silent),
                (true, End::Open { .. }) => None,
            };
            let weight = self.cum[i1] - self.cum[i0];
            let r = speech.filter(|s| *s > 0.0 && weight > 0.0).map_or(rate, |s| weight / s);
            for (start, before) in starts[i0..i1].iter_mut().zip(&self.cum[i0..i1]) {
                let mut t = resumes + (before - self.cum[i0]) / r.max(1e-6);
                for p in inner {
                    if p.start <= t {
                        t += p.len();
                    }
                }
                *start = t;
            }
        }
        starts
    }

    /// The rate the matched stretches were spoken at; `None` with under a second of them.
    fn learnt_rate(&self, pairs: &[(usize, usize)]) -> Option<f64> {
        let (mut weight, mut secs) = (0.0, 0.0);
        let (mut i0, mut resumes, mut m0) = (0, self.start, 0);
        for &(k, m) in pairs {
            let i1 = self.breaks[k].0 + 1;
            weight += self.cum[i1] - self.cum[i0];
            secs += self.pauses[m].start - resumes - (self.silence[m] - self.silence[m0]);
            (i0, resumes, m0) = (i1, self.pauses[m].end, m + 1);
        }
        (secs > 1.0).then(|| weight / secs)
    }
}

/// `[0, x0, x0 + x1, ...]`.
fn running(xs: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut sums = vec![0.0];
    for x in xs {
        sums.push(sums[sums.len() - 1] + x);
    }
    sums
}

/// The sum of the items strictly between `from` and `to`, numbered from 1, of a running sum.
fn between(sums: &[f64], from: usize, to: usize) -> f64 {
    let n = sums.len() - 1;
    let (a, b) = (from.min(n), to.saturating_sub(1).min(n));
    if b > a { sums[b] - sums[a] } else { 0.0 }
}

/// English numbers as a voice reads them, in syllables.
mod numbers {
    const ONES: [u32; 10] = [0, 1, 1, 1, 1, 1, 1, 2, 1, 1];
    const TEENS: [u32; 10] = [1, 3, 1, 2, 2, 2, 2, 3, 2, 2];
    const TENS: [u32; 10] = [0, 0, 2, 2, 2, 2, 2, 3, 2, 2];
    const EDGES: &[char] =
        &['.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '"', '\'', '“', '”', '‘', '’'];

    /// A price, a quantity or a year read as a number, an ID digit by digit; `None` otherwise.
    pub(super) fn spoken(word: &str) -> Option<f64> {
        let t = word.trim_matches(EDGES);
        if let Some(price) = t.strip_prefix('$') {
            let (dollars, cents) =
                price.split_once('.').map_or((price, None), |(d, c)| (d, Some(c)));
            let digits =
                |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit() || c == ',');
            if dollars.starts_with(|c: char| c.is_ascii_digit())
                && digits(dollars)
                && cents.is_none_or(|c| c.len() == 2 && c.chars().all(|c| c.is_ascii_digit()))
            {
                let mut s = cardinal(dollars.replace(',', "").parse().ok()?) + 2;
                if let Some(c) = cents.and_then(|c| c.parse::<u64>().ok()).filter(|&c| c > 0) {
                    s += 1 + cardinal(c) + 1;
                }
                return Some(f64::from(s));
            }
        }
        if is_quantity(t) && !(t.len() > 1 && t.starts_with('0')) {
            let plain = t.replace(',', "");
            let (whole, frac) = plain.split_once('.').map_or((plain.as_str(), ""), |(w, f)| (w, f));
            let n: u64 = whole.parse().ok()?;
            let mut s = if whole.len() == 4 && (1900..=2099).contains(&n) {
                below_1000(n / 100) + below_1000(n % 100)
            } else {
                cardinal(n)
            };
            if !frac.is_empty() {
                s += 2 + frac
                    .chars()
                    .filter_map(|c| c.to_digit(10))
                    .map(|d| match ONES[d as usize] {
                        0 => 2,
                        n => n,
                    })
                    .sum::<u32>();
            }
            return Some(f64::from(s));
        }
        if is_id(t) {
            let s = t.chars().filter(|c| c.is_alphanumeric()).map(|c| match c.to_digit(10) {
                Some(d) => match ONES[d as usize] {
                    0 => 2,
                    n => n,
                },
                None if c == 'W' => 3,
                None => 1,
            });
            return Some(f64::from(s.sum::<u32>()));
        }
        None
    }

    /// `\d{1,4}(,\d{3})*(\.\d+)?`
    fn is_quantity(t: &str) -> bool {
        let (int, frac) = t.split_once('.').map_or((t, None), |(i, f)| (i, Some(f)));
        if frac.is_some_and(|f| f.is_empty() || !f.chars().all(|c| c.is_ascii_digit())) {
            return false;
        }
        let mut groups = int.split(',');
        let first = groups.next().unwrap_or("");
        (1..=4).contains(&first.len())
            && first.chars().all(|c| c.is_ascii_digit())
            && groups.all(|g| g.len() == 3 && g.chars().all(|c| c.is_ascii_digit()))
    }

    /// `#?[A-Z]*\d[\dA-Z-]*`
    fn is_id(t: &str) -> bool {
        let t = t.strip_prefix('#').unwrap_or(t);
        let rest = t.trim_start_matches(|c: char| c.is_ascii_uppercase());
        rest.starts_with(|c: char| c.is_ascii_digit())
            && rest.chars().all(|c| c.is_ascii_digit() || c.is_ascii_uppercase() || c == '-')
    }

    fn below_1000(mut n: u64) -> u32 {
        let mut s = 0;
        if n >= 100 {
            s += ONES[(n / 100) as usize] + 2;
            n %= 100;
        }
        if (10..20).contains(&n) {
            s += TEENS[(n - 10) as usize];
        } else {
            s += TENS[(n / 10) as usize] + ONES[(n % 10) as usize];
        }
        s
    }

    fn cardinal(mut n: u64) -> u32 {
        if n == 0 {
            return 2;
        }
        let mut s = 0;
        for (scale, words) in [(1_000_000_000, 3), (1_000_000, 2), (1_000, 2)] {
            if n >= scale {
                s += below_1000(n / scale) + words;
                n %= scale;
            }
        }
        s + if n > 0 { below_1000(n) } else { 0 }
    }
}

#[cfg(test)]
#[path = "estimate_tests.rs"]
mod tests;
