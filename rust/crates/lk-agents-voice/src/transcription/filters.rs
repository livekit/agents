use lk_agents_abi::tokenize::is_py_space;

/// `INLINE_SPLIT_TOKENS`: the buffer is cut after the last of these.
const SPLIT_TOKENS: [char; 11] = [' ', ',', '.', '?', '!', ';', '，', '。', '？', '！', '；'];

/// Scripts that put an emphasis delimiter flush against a word (CJK, kana, Thai, Korean particles).
fn is_flush_emphasis_script(c: char) -> bool {
    matches!(c,
        '\u{0e00}'..='\u{0e7f}'   // thai
        | '\u{1100}'..='\u{11ff}' // hangul jamo
        | '\u{3040}'..='\u{30ff}' // hiragana, katakana
        | '\u{3130}'..='\u{318f}' // hangul compatibility jamo
        | '\u{3400}'..='\u{4dbf}' // cjk unified ideographs extension a
        | '\u{4e00}'..='\u{9fff}' // cjk unified ideographs
        | '\u{ac00}'..='\u{d7af}' // hangul syllables
        | '\u{f900}'..='\u{faff}' // cjk compatibility ideographs
        | '\u{ff66}'..='\u{ff9d}' // halfwidth katakana
    )
}

/// Python's `\w` for `str`: alphanumeric or underscore, Unicode-aware.
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `_INTRAWORD`: what makes an asterisk intra-word, so `2*3*4` isn't emphasis.
fn is_intraword(c: char) -> bool {
    is_word(c) && !is_flush_emphasis_script(c)
}

/// `INLINE_MARKERS`: text with none of these matches no inline pattern.
fn has_inline_marker(s: &str) -> bool {
    s.chars().any(|c| matches!(c, '*' | '_' | '`' | '~' | '['))
}

/// Python's `re.sub` over a whole string; `try_match(chars, i)` returns `(end, replacement)`.
fn sub_all(
    s: &str,
    mut try_match: impl FnMut(&[char], usize) -> Option<(usize, String)>,
) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if let Some((end, replacement)) = try_match(&chars, i) {
            out.push_str(&replacement);
            i = end;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// `!\[([^\]]*)\]\([^)]*\)` -> `\1`, then `\[([^\]]*)\]\([^)]*\)` -> `\1`; images first so the
/// alt text keeps no stray `!`.
fn sub_links(s: &str, image: bool) -> String {
    sub_all(s, |c, i| {
        let mut p = i;
        if image {
            if c.get(p) != Some(&'!') {
                return None;
            }
            p += 1;
        }
        if c.get(p) != Some(&'[') {
            return None;
        }
        p += 1;
        let text_start = p;
        while p < c.len() && c[p] != ']' {
            p += 1;
        }
        if p >= c.len() {
            return None;
        }
        let text: String = c[text_start..p].iter().collect();
        p += 1;
        if c.get(p) != Some(&'(') {
            return None;
        }
        p += 1;
        while p < c.len() && c[p] != ')' {
            p += 1;
        }
        if p >= c.len() {
            return None;
        }
        Some((p + 1, text))
    })
}

/// `(?<!INTRAWORD)(?<!\*)(\*{1,3})(?!\s)([^*\n]+?)(?<!\s)\1(?!INTRAWORD)(?!\*)` -> `\2`, and the
/// underscore twin `(?<!\w)(_{1,3})(?!\s)([^_\n]+?)(?<!\s)\1(?!\w)`, whose full `\w` boundary
/// protects `snake_case` in every script. The body holds no delimiter, so a scan suffices.
fn sub_emphasis(s: &str, delim: char, boundary: fn(char) -> bool) -> String {
    sub_all(s, |c, i| {
        if c[i] != delim {
            return None;
        }
        if i > 0 && (boundary(c[i - 1]) || c[i - 1] == delim) {
            return None;
        }
        let mut run = 0;
        while i + run < c.len() && c[i + run] == delim {
            run += 1;
        }
        // A run of four or more can never open emphasis.
        if run > 3 {
            return None;
        }
        let open_end = i + run;
        // `(?!\s)` plus the body's own character class.
        if open_end >= c.len() || is_py_space(c[open_end]) {
            return None;
        }
        let mut body_end = open_end;
        while body_end < c.len() && c[body_end] != delim && c[body_end] != '\n' {
            body_end += 1;
        }
        if body_end == open_end {
            return None; // the body needs at least one character
        }
        if is_py_space(c[body_end - 1]) {
            return None; // `(?<!\s)` before the closing run
        }
        let mut close = 0;
        while body_end + close < c.len() && c[body_end + close] == delim {
            close += 1;
        }
        if close < run {
            return None;
        }
        let close_end = body_end + run;
        if let Some(&after) = c.get(close_end)
            && (boundary(after) || after == delim)
        {
            return None;
        }
        Some((close_end, c[open_end..body_end].iter().collect()))
    })
}

/// `` `{3,4}[\S]* `` -> `""`. A fence and its info string carry no spoken content.
fn sub_code_fences(s: &str) -> String {
    sub_all(s, |c, i| {
        if c[i] != '`' {
            return None;
        }
        let mut ticks = 0;
        while i + ticks < c.len() && c[i + ticks] == '`' && ticks < 4 {
            ticks += 1;
        }
        if ticks < 3 {
            return None;
        }
        let mut p = i + ticks;
        while p < c.len() && !is_py_space(c[p]) {
            p += 1;
        }
        Some((p, String::new()))
    })
}

/// `` `([^`]+?)` `` -> `\1`.
fn sub_inline_code(s: &str) -> String {
    sub_all(s, |c, i| {
        if c[i] != '`' {
            return None;
        }
        let mut p = i + 1;
        while p < c.len() && c[p] != '`' {
            p += 1;
        }
        if p >= c.len() || p == i + 1 {
            return None; // unterminated, or an empty body the `+?` forbids
        }
        Some((p + 1, c[i + 1..p].iter().collect()))
    })
}

/// `~~(?!\s)[^~]*?(?<!\s)~~` -> `""`. Struck text is not spoken, so the body goes with the marks.
fn sub_strikethrough(s: &str) -> String {
    sub_all(s, |c, i| {
        if c[i] != '~' || c.get(i + 1) != Some(&'~') {
            return None;
        }
        let start = i + 2;
        if start >= c.len() || is_py_space(c[start]) {
            return None;
        }
        let mut body_end = start;
        while body_end < c.len() && c[body_end] != '~' {
            body_end += 1;
        }
        if c.get(body_end) != Some(&'~') || c.get(body_end + 1) != Some(&'~') {
            return None;
        }
        if is_py_space(c[body_end - 1]) {
            return None;
        }
        Some((body_end + 2, String::new()))
    })
}

/// `^ {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$`. Only for complete lines: a
/// prefix of `____ and so on` looks like a rule.
fn is_horizontal_rule(line: &str) -> bool {
    let c: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < c.len() && i < 3 && c[i] == ' ' {
        i += 1;
    }
    let Some(&marker) = c.get(i) else { return false };
    if !matches!(marker, '-' | '*' | '_') {
        return false;
    }
    let mut count = 0;
    while i < c.len() {
        if c[i] != marker {
            return false;
        }
        count += 1;
        i += 1;
        while i < c.len() && (c[i] == ' ' || c[i] == '\t') {
            i += 1;
        }
    }
    count >= 3
}

/// The three `LINE_PATTERNS`, sequentially as the reference's three `sub` passes.
fn strip_line_markers(line: &str) -> String {
    let mut out = line.to_owned();

    // `^#{1,6}\s+`
    {
        let c: Vec<char> = out.chars().collect();
        let hashes = c.iter().take_while(|&&x| x == '#').count();
        if (1..=6).contains(&hashes) && c.get(hashes).is_some_and(|&x| is_py_space(x)) {
            let mut end = hashes;
            while end < c.len() && is_py_space(c[end]) {
                end += 1;
            }
            out = c[end..].iter().collect();
        }
    }
    // `^\s*[-+*]\s+`
    out = strip_leading_marker(&out, |c| matches!(c, '-' | '+' | '*'));
    // `^\s*>\s+`
    out = strip_leading_marker(&out, |c| c == '>');
    out
}

fn strip_leading_marker(line: &str, is_marker: impl Fn(char) -> bool) -> String {
    let c: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < c.len() && is_py_space(c[i]) {
        i += 1;
    }
    if !c.get(i).copied().is_some_and(&is_marker) {
        return line.to_owned();
    }
    let mut end = i + 1;
    let space_start = end;
    while end < c.len() && is_py_space(c[end]) {
        end += 1;
    }
    if end == space_start {
        return line.to_owned(); // `\s+` needs at least one
    }
    c[end..].iter().collect()
}

/// `process_complete_text`. `at_line_start` enables the line patterns; `line_complete` the
/// whole-line ones.
pub fn process_complete_text(text: &str, at_line_start: bool, line_complete: bool) -> String {
    let mut out = text.to_owned();
    if at_line_start {
        if line_complete && is_horizontal_rule(&out) {
            out = String::new();
        }
        out = strip_line_markers(&out);
    }
    if !has_inline_marker(&out) {
        return out;
    }
    out = sub_links(&out, true);
    out = sub_links(&out, false);
    // Emphasis before code and strikethrough (`**~~text~~**`); asterisks get a second pass for
    // `_**x**_`.
    out = sub_emphasis(&out, '*', is_intraword);
    out = sub_emphasis(&out, '_', is_word);
    out = sub_emphasis(&out, '*', is_intraword);
    out = sub_code_fences(&out);
    out = sub_inline_code(&out);
    out = sub_strikethrough(&out);
    out
}

/// Non-overlapping occurrences of `needle` in `haystack`. Python's `str.count`.
fn count_str(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

/// Whether a delimiter's occurrences cannot pair up yet.
fn unbalanced(buffer: &str, delimiter: char) -> bool {
    let double = format!("{delimiter}{delimiter}");
    let doubles = count_str(buffer, &double);
    if doubles % 2 == 1 {
        return true;
    }
    (buffer.chars().filter(|&c| c == delimiter).count() - doubles * 2) % 2 == 1
}

/// `\[[^\]]*\]\([^)]*\)`, or the same with a leading `!`. Counted, not extracted.
fn count_links(s: &str, image: bool) -> usize {
    let c: Vec<char> = s.chars().collect();
    let (mut i, mut found) = (0, 0);
    while i < c.len() {
        let mut p = i;
        let ok = (|| {
            if image {
                if c.get(p) != Some(&'!') {
                    return false;
                }
                p += 1;
            }
            if c.get(p) != Some(&'[') {
                return false;
            }
            p += 1;
            while p < c.len() && c[p] != ']' {
                p += 1;
            }
            if p >= c.len() {
                return false;
            }
            p += 1;
            if c.get(p) != Some(&'(') {
                return false;
            }
            p += 1;
            while p < c.len() && c[p] != ')' {
                p += 1;
            }
            if p >= c.len() {
                return false;
            }
            p += 1;
            true
        })();
        if ok {
            found += 1;
            i = p;
        } else {
            i += 1;
        }
    }
    found
}

/// Whether the buffer might hold an unfinished construct. Conservative: a false "no" publishes a
/// half-parsed `*`.
pub fn has_incomplete_pattern(buffer: &str) -> bool {
    if buffer
        .chars()
        .next_back()
        .is_some_and(|c| matches!(c, '#' | '-' | '+' | '*' | '_' | '>' | '!' | '`' | '~' | ' '))
    {
        return true;
    }
    if unbalanced(buffer, '*') || unbalanced(buffer, '_') {
        return true;
    }
    if buffer.chars().filter(|&c| c == '`').count() % 2 == 1 || count_str(buffer, "~~") % 2 == 1 {
        return true;
    }
    let open_brackets = buffer.chars().filter(|&c| c == '[').count();
    // An image matches the link pattern too (the `!` is not part of it), so both are subtracted.
    open_brackets as i64 - count_links(buffer, false) as i64 - count_links(buffer, true) as i64 > 0
}

/// `filter_markdown` as a push accumulator. Returns the pieces per chunk, not joined: the
/// synchroniser paces per chunk.
#[lk_agents_abi::class(mutable)]
#[derive(Debug, Default)]
pub struct MarkdownFilter {
    buffer: String,
    at_line_start: bool,
}

#[lk_agents_abi::host(object)]
impl MarkdownFilter {
    /// At line start (the `Default` starts mid-line, which would miss a leading `# `).
    #[host(new)]
    pub fn new() -> Self {
        MarkdownFilter { buffer: String::new(), at_line_start: true }
    }

    #[host(method)]
    pub fn push(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.push_str(chunk);

        if self.buffer.contains('\n') {
            let lines: Vec<String> = self.buffer.split('\n').map(str::to_owned).collect();
            let last = lines.len() - 1;
            self.buffer = lines[last].clone();
            let mut out = Vec::new();
            for (i, line) in lines[..last].iter().enumerate() {
                let at_line_start = if i == 0 { self.at_line_start } else { true };
                out.push(process_complete_text(line, at_line_start, true) + "\n");
            }
            self.at_line_start = true;
            return out;
        }

        // The last split token in the buffer; everything before it is a candidate for release.
        let chars: Vec<char> = self.buffer.chars().collect();
        let Some(cut) = chars.iter().rposition(|c| SPLIT_TOKENS.contains(c)) else {
            return Vec::new();
        };
        // `>= 1`: a buffer that *begins* with a split token has nothing before it to release.
        if cut < 1 {
            return Vec::new();
        }
        let processable: String = chars[..cut].iter().collect();
        if has_incomplete_pattern(&processable) {
            return Vec::new();
        }
        let rest: String = chars[cut..].iter().collect();
        let out = process_complete_text(&processable, self.at_line_start, false);
        self.buffer = rest;
        self.at_line_start = false;
        vec![out]
    }

    /// End of stream: release the tail, which is now known to be a complete line.
    #[host(method)]
    pub fn flush(&mut self) -> Vec<String> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let out = process_complete_text(&self.buffer, self.at_line_start, true);
        self.buffer.clear();
        vec![out]
    }
}

/// `EMOJI_PATTERN` as a predicate (its trailing `+` binds only to the keycap combiner).
fn is_emoji(c: char) -> bool {
    matches!(c,
        '\u{1F000}'..='\u{1FBFF}'   // Mahjong Tiles .. Symbols for Legacy Computing
        | '\u{2600}'..='\u{26FF}'   // Miscellaneous Symbols
        | '\u{2700}'..='\u{27BF}'   // Dingbats
        | '\u{2B00}'..='\u{2BFF}'   // Miscellaneous Symbols and Arrows
        | '\u{FE00}'..='\u{FE0F}'   // Variation selectors
        | '\u{200D}'                // Zero width joiner
        | '\u{20E3}'                // Combining enclosing keycap
    )
}

/// `filter_emoji`. Stateless: nothing here spans a chunk boundary.
#[lk_agents_abi::host(export(pid = false))]
pub fn filter_emoji(chunk: &str) -> String {
    if !chunk.chars().any(is_emoji) {
        return chunk.to_owned();
    }
    chunk.chars().filter(|&c| !is_emoji(c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: `Default::default()` starting with `at_line_start: false`, so a first-chunk `# `
    // isn't a heading.
    #[test]
    fn the_filter_starts_at_a_line_start() {
        let mut f = MarkdownFilter::new();
        let mut out = f.push("# Heading\nbody");
        out.extend(f.flush());
        let joined = out.concat();
        assert!(!joined.contains('#'), "a leading heading marker must be stripped: {joined:?}");
        assert!(joined.contains("Heading") && joined.contains("body"));

        let mut d = MarkdownFilter::default();
        let mut kept = d.push("# Heading\nbody");
        kept.extend(d.flush());
        assert!(kept.concat().contains('#'), "`Default` is mid-line, and that is the difference");
    }

    // Catches: `push` joining the pieces (destroying pacing), or releasing a `*` that may still open.
    #[test]
    fn push_returns_pieces_and_holds_back_an_incomplete_pattern() {
        let mut f = MarkdownFilter::new();
        let held = f.push("plain *");
        assert!(held.iter().all(|p| !p.ends_with('*')), "an open pattern must not be emitted");
        let rest = f.flush();
        assert!(!rest.is_empty(), "flush must release what push held");
        assert_eq!(held.concat() + &rest.concat(), "plain *", "nothing is lost across the two");
    }
}
