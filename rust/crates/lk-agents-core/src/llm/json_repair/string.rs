//! Translation of parse_string.py and parse_string_helpers from the attributed upstream.
use super::*;
use std::ops::ControlFlow;
impl Parser {
    fn member(&self, mut i: isize) -> bool {
        if self.at(i).is_some_and(quote) {
            let q = closing(self.at(i).unwrap());
            i = self.skip(&[q], i + 1);
            if self.at(i).is_none() {
                return false;
            }
            i = self.scroll(i + 1);
            return self.at(i) == Some(':');
        }
        if !self.at(i).is_some_and(|c| c.is_alphanumeric() || c == '_') {
            return false;
        }
        while self.at(i).is_some_and(|c| c.is_alphanumeric() || matches!(c, '_' | '-')) {
            i += 1;
        }
        self.at(self.scroll(i)) == Some(':')
    }
    fn comma_kind(&self) -> Comma {
        let i = self.scroll(1);
        match self.at(i) {
            None | Some('}') => return Comma::EndsValue,
            Some('{' | '[') => return Comma::OpensContainer,
            _ => {}
        }
        if self.at(i).is_some_and(quote) {
            let q = self.at(i).unwrap();
            let end = self.skip(&[q], i + 1);
            return if self.at(end).is_some() && self.at(self.scroll(end + 1)) == Some(':') {
                Comma::EndsValue
            } else {
                Comma::InText
            };
        }
        if self.member(if self.at(i) == Some('`') { i + 1 } else { i }) {
            return Comma::EndsValue;
        }
        let q = self.skip(&['"', '\'', '“', '”'], i);
        let c = self.skip(&['{', '['], i);
        if self.at(c).is_some() && c < q {
            return Comma::InText;
        }
        if self.at(i).is_some_and(quote) {
            return Comma::InText;
        }
        if self.member(q) { Comma::EndsValue } else { Comma::InText }
    }
    pub(super) fn string(&mut self) -> Value {
        if matches!(self.at(0), Some('#' | '/')) {
            return self.comment();
        }
        while self.at(0).is_some_and(|c| !quote(c) && !c.is_alphanumeric()) {
            self.index += 1;
        }
        let Some(first) = self.at(0) else {
            return empty();
        };
        let left = if matches!(first, '\'' | '“') { first } else { '"' };
        let right = closing(left);
        let missing = first.is_alphanumeric();
        if missing && self.current() != Some(Context::Key) {
            for (word, value) in
                [("true", Value::Bool(true)), ("false", Value::Bool(false)), ("null", Value::Null)]
            {
                if self.slice(self.index, self.index + word.len()).to_lowercase() == word {
                    self.index += word.len();
                    return value;
                }
            }
        }
        // Upstream's fast path only accepts quotes followed by a context-appropriate delimiter.
        if first == '"' {
            let i = self.skip(&['"'], 1);
            let text = self.slice(self.index + 1, self.index + i as usize);
            let next = self.at(self.scroll(i + 1));
            let valid = match self.current() {
                Some(Context::Key) => next == Some(':'),
                Some(Context::Value) => matches!(next, None | Some(',' | '}')),
                Some(Context::Array) => matches!(next, None | Some(',' | ']')),
                None => next.is_none(),
            };
            if self.at(i).is_some() && !text.contains(['\\', '\n', '\r']) && valid {
                self.index += i as usize + 1;
                return text.into();
            }
        }
        if !missing {
            self.index += 1;
        }
        if self.slice(self.index, self.index + 7) == "```json" {
            let i = self.skip(&['`'], 7);
            if self.slice(self.index + i as usize, self.index + i as usize + 3) == "```" {
                self.index += 7;
                return self.json();
            }
        }

        let mut doubled = false;
        if self.at(0) == Some(left) {
            if match self.current() {
                Some(Context::Key) => self.at(1) == Some(':'),
                Some(Context::Value) => matches!(self.at(1), Some(',' | '}')),
                Some(Context::Array) => matches!(self.at(1), Some(',' | ']')),
                None => false,
            } {
                self.index += 1;
                return empty();
            }
            if self.at(1) == Some(left) {
                return empty();
            }
            let i = self.skip(&[right], 1);
            if self.at(i + 1) == Some(right) {
                doubled = true;
                self.index += 1;
            } else {
                let next = self.at(self.scroll(1));
                if next.is_some_and(|c| quote(c) || matches!(c, '{' | '[')) {
                    self.index += 1;
                    return empty();
                }
                if !matches!(next, Some(',' | ']' | '}')) {
                    self.index += 1;
                }
            }
        }
        let mut scan = StringScan {
            left,
            right,
            missing,
            doubled,
            out: String::new(),
            unmatched: false,
            low: false,
            pending: false,
            stack: Vec::new(),
        };
        while let Some(c) = self.at(0) {
            if self.string_step(&mut scan, c).is_break() {
                break;
            }
        }
        self.finish_string(scan)
    }
    /// One character of a string's body. The cases run in upstream's order; `Continue` moves on
    /// to the next character, `Break` ends the string.
    fn string_step(&mut self, s: &mut StringScan, c: char) -> ControlFlow<()> {
        if c == s.right && !s.low || self.bare_word_ends(s, c) {
            return ControlFlow::Break(());
        }
        if c == '„' && !s.out.ends_with('\\') || s.low && c == '”' {
            s.low = c == '„';
            return s.keep(self, c);
        }
        if s.pending
            && matches!(c, '[' | '{' | '(')
            && let Some(end) = self.inline_end(0)
        {
            s.out.push_str(&self.slice(self.index, self.index + end as usize));
            self.index += end as usize;
            s.pending = false;
            s.stack.clear();
            return ControlFlow::Continue(());
        }
        if let Some(flow) = self.value_comma(s, c) {
            return flow;
        }
        s.track_nesting(c);
        if matches!((c, s.stack.last()), ('}', Some('{')) | (']', Some('['))) {
            s.stack.pop();
            return s.keep(self, c);
        }
        if let Some(flow) = self.brace_in_value(s, c) {
            return flow;
        }
        if c == ']'
            && self.has(Context::Array)
            && !s.out.ends_with(s.right)
            && self.at(self.skip(&[s.right], 0)).is_none()
        {
            return ControlFlow::Break(());
        }
        if self.fenced_value_ends(s, c) {
            return ControlFlow::Break(());
        }
        s.out.push(c);
        self.index += 1;
        if let Some(flow) = self.escape(s) {
            return flow;
        }
        if self.colon_ends_key(s) {
            return ControlFlow::Break(());
        }
        if s.low && self.at(0) == Some('"') {
            s.low = false;
            return s.keep(self, '"');
        }
        self.closing_quote(s).unwrap_or(ControlFlow::Continue(()))
    }
    /// An unquoted key ends at a colon or space, an unquoted array item at `]` or `,`.
    fn bare_word_ends(&self, s: &StringScan, c: char) -> bool {
        s.missing
            && ((self.current() == Some(Context::Key) && (c == ':' || c.is_whitespace()))
                || (self.current() == Some(Context::Array) && matches!(c, ']' | ',')))
    }
    /// A comma in a value: it ends the value, opens a container, or is text.
    fn value_comma(&mut self, s: &mut StringScan, c: char) -> Option<ControlFlow<()>> {
        if self.current() != Some(Context::Value) || c != ',' || s.pending || !s.stack.is_empty() {
            return None;
        }
        match self.comma_kind() {
            Comma::EndsValue => return Some(ControlFlow::Break(())),
            Comma::OpensContainer => s.pending = true,
            Comma::InText => {}
        }
        Some(s.keep(self, c))
    }
    /// A `}` in a value: text when it closes a `{` of the text, else possibly the object's end.
    fn brace_in_value(&mut self, s: &mut StringScan, c: char) -> Option<ControlFlow<()>> {
        if self.current() != Some(Context::Value) || c != '}' || s.out.ends_with(s.right) {
            return None;
        }
        if opens_literal_brace(&s.out) {
            return Some(s.keep(self, c));
        }
        self.brace_ends_value(s).then_some(ControlFlow::Break(()))
    }
    /// Whether the `}` ends a value whose closing quote is missing.
    fn brace_ends_value(&self, s: &StringScan) -> bool {
        let escaped = self.at(1) == Some('\\');
        let i = self.skip(&[s.right], 1);
        if self.at(i).is_none() {
            if self.at(self.skip(&[':'], 1)).is_some() {
                return true;
            }
            let i = self.scroll(1);
            // more than one character before the next `}` means the text goes on
            return !escaped && self.skip(&['}'], i) - i <= 1;
        }
        if escaped {
            return false;
        }
        let i = self.scroll(i + 1);
        if matches!(self.at(i), None | Some(',' | '}')) {
            return false;
        }
        let i = self.skip(&[s.left], i);
        if self.at(i).is_none() {
            return false;
        }
        !self.at(self.scroll(i + 1)).is_some_and(|c| c != ':')
    }
    /// A `}` in a value at the end of input, or before a code fence the value does not run
    /// past. A container right after the fence is kept as text.
    fn fenced_value_ends(&self, s: &mut StringScan, c: char) -> bool {
        if self.current() != Some(Context::Value) || c != '}' {
            return false;
        }
        let i = self.scroll(1);
        if self.at(i).is_none() {
            return true;
        }
        if !FENCE.chars().zip(i..).all(|(f, j)| self.at(j) == Some(f)) {
            return false;
        }
        let mut start = self.comment_scroll(i + FENCE.len() as isize);
        let mut keep_container = false;
        if self.at(start).is_some_and(|c| matches!(c, '[' | '{' | '('))
            && let Some(end) = self.inline_end(start)
        {
            let after = self.scroll(end);
            let ch = self.at(after);
            let next = self.comment_scroll(after + 1);
            if matches!(ch, None | Some('}'))
                || ch == Some(',') && (self.at(next).is_none_or(|c| c == '}') || self.member(next))
            {
                return true;
            }
            keep_container = true;
            start = end;
        }
        let mut q = self.skip(&[s.right], start);
        while self.at(q).is_some() {
            if matches!(self.at(self.scroll(q + 1)), None | Some(',' | '}' | ']')) {
                break;
            }
            q = self.skip(&[s.right], q + 1);
        }
        if self.at(q).is_none() {
            return true;
        }
        if keep_container {
            s.pending = true;
        }
        false
    }
    /// A backslash just kept, followed by an escape: the pair becomes the character it names.
    fn escape(&mut self, s: &mut StringScan) -> Option<ControlFlow<()>> {
        if !s.out.ends_with('\\') {
            return None;
        }
        let e = self.at(0)?;
        if !(matches!(e, 't' | 'n' | 'r' | 'b' | '\\') || e == s.right || quote(e) || e == '„') {
            return self.unicode_escape(s, e);
        }
        s.out.pop();
        s.out.push(match e {
            't' => '\t',
            'n' => '\n',
            'r' => '\r',
            'b' => '\u{8}',
            _ => e,
        });
        self.index += 1;
        while s.out.ends_with('\\') && self.at(0).is_some_and(|c| c == s.right || c == '\\') {
            s.out.pop();
            s.out.push(self.at(0).unwrap());
            self.index += 1;
        }
        Some(ControlFlow::Continue(()))
    }
    /// `\uXXXX` (a surrogate pair when two follow) or `\xXX`.
    fn unicode_escape(&mut self, s: &mut StringScan, e: char) -> Option<ControlFlow<()>> {
        let n = match e {
            'u' => UNICODE_ESCAPE_DIGITS,
            'x' => HEX_ESCAPE_DIGITS,
            _ => return None,
        };
        let code = self.slice(self.index + 1, self.index + 1 + n);
        if code.len() != n || !code.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let u = u32::from_str_radix(&code, 16).unwrap();
        if HIGH_SURROGATES.contains(&u)
            && self.at((n + 1) as isize) == Some('\\')
            && self.at((n + 2) as isize) == Some('u')
        {
            let low_at = self.index + 1 + n + "\\u".len();
            let low_end = low_at + UNICODE_ESCAPE_DIGITS;
            if let Ok(low) = u32::from_str_radix(&self.slice(low_at, low_end), 16)
                && let Some(ch) = surrogate_pair(u, low)
            {
                s.out.pop();
                s.out.push(ch);
                self.index = low_end;
                return Some(ControlFlow::Continue(()));
            }
        }
        let ch = char::from_u32(u)?;
        s.out.pop();
        s.out.push(ch);
        self.index += 1 + n;
        Some(ControlFlow::Continue(()))
    }
    /// In a quoted key, a colon ends the key unless a quoted key-value pair follows it.
    fn colon_ends_key(&self, s: &StringScan) -> bool {
        if self.at(0) != Some(':') || s.missing || self.current() != Some(Context::Key) {
            return false;
        }
        let i = self.skip(&[s.left], 1);
        if self.at(i).is_none() {
            return true;
        }
        let i = self.skip(&[s.right], i + 1);
        self.at(i).is_some() && matches!(self.at(self.scroll(i + 1)), Some(',' | '}'))
    }
    /// The closing quote character: the string's end, or a quote inside its text.
    fn closing_quote(&mut self, s: &mut StringScan) -> Option<ControlFlow<()>> {
        if self.at(0) != Some(s.right) || s.out.ends_with('\\') {
            return None;
        }
        if s.doubled && self.at(1) == Some(s.right) {
            self.index += 1;
            return Some(ControlFlow::Continue(()));
        }
        if s.missing && self.current() == Some(Context::Value) {
            let i = self.skip(&[s.right, s.left], 1);
            if self.at(i).is_some() && self.at(self.scroll(i + 1)) == Some(':') {
                self.index = self.index.saturating_sub(1);
                return Some(ControlFlow::Break(()));
            }
            return Some(ControlFlow::Continue(()));
        }
        if s.unmatched {
            s.unmatched = false;
            return Some(s.keep(self, s.right));
        }
        let i = self.quote_lookahead(s);
        let next = self.at(i);
        if next == Some(',') && self.current() == Some(Context::Value) {
            let i = self.skip(&[s.right], i + 1);
            if matches!(self.at(self.scroll(i + 1)), Some('}' | ',')) {
                return Some(s.keep(self, s.right));
            }
            return None;
        }
        if next != Some(s.right) || self.at(i - 1) == Some('\\') {
            return None;
        }
        if (1..i).all(|j| self.at(j).is_none_or(char::is_whitespace))
            && !(self.current() == Some(Context::Value) && self.member_follows_comma(i))
        {
            return Some(ControlFlow::Break(()));
        }
        match self.current() {
            Some(Context::Value) => self.quote_in_value(s, i),
            Some(Context::Array) => Some(self.quote_in_array(s, i)),
            Some(Context::Key) => Some(s.keep(self, s.right)),
            None => None,
        }
    }
    /// The offset after a quote where a quote, or a delimiter of the current context, stops it.
    fn quote_lookahead(&self, s: &StringScan) -> isize {
        let mut i = 1;
        let mut comma_ends_value = true;
        while self.at(i).is_some_and(|c| c != s.right && c != s.left) {
            let ch = self.at(i).unwrap();
            if comma_ends_value && ch.is_alphabetic() {
                comma_ends_value = false;
            }
            if self.has(Context::Key) && matches!(ch, ':' | '}')
                || self.has(Context::Value) && ch == '}'
                || self.has(Context::Array) && matches!(ch, ']' | ',')
                || comma_ends_value && self.current() == Some(Context::Value) && ch == ','
            {
                break;
            }
            i += 1;
        }
        i
    }
    /// After the quote at offset `j`, a comma and then another member.
    fn member_follows_comma(&self, j: isize) -> bool {
        let c = self.scroll(j + 1);
        self.at(c) == Some(',') && self.member(self.comment_scroll(c + 1))
    }
    /// A quote in a value followed by another quote at `i`: text unless a key follows.
    fn quote_in_value(&mut self, s: &mut StringScan, i: isize) -> Option<ControlFlow<()>> {
        if self.member_follows_comma(i) {
            return Some(s.keep(self, s.right));
        }
        let mut i = self.skip(&[s.right], i + 1) + 1;
        while self.at(i).is_some_and(|c| c != ':') {
            let ch = self.at(i).unwrap();
            if matches!(ch, ',' | ']' | '}') || (ch == s.right && self.at(i - 1) != Some('\\')) {
                break;
            }
            i += 1;
        }
        if self.at(i) == Some(':') {
            return None;
        }
        s.unmatched = !s.unmatched;
        Some(s.keep(self, s.right))
    }
    /// A quote in an array item: text when the quotes after it pair up, else the item's end.
    fn quote_in_array(&mut self, s: &mut StringScan, mut i: isize) -> ControlFlow<()> {
        while self.at(i) == Some(s.right) {
            i = self.skip(&[s.right, ']'], i + 1);
            if self.at(i) != Some(s.right) {
                return ControlFlow::Break(());
            }
            i = self.skip(&[s.right, ']'], i + 1);
        }
        s.unmatched = !s.unmatched;
        s.keep(self, s.right)
    }
    /// After the body: an unquoted key not followed by `:` or `,` is no key; trailing space of an
    /// unterminated or unquoted string is dropped.
    fn finish_string(&mut self, s: StringScan) -> Value {
        let mut out = s.out;
        if s.missing
            && self.current() == Some(Context::Key)
            && self.at(0).is_some_and(char::is_whitespace)
        {
            self.ws();
            if !matches!(self.at(0), Some(':' | ',')) {
                return empty();
            }
        }
        if self.at(0) != Some(s.right) {
            out = out.trim_end().to_owned();
        } else {
            self.index += 1;
        }
        if s.missing || out.ends_with('\n') {
            out = out.trim_end().to_owned();
        }
        Value::String(out)
    }
}

/// What a comma inside a value's text means.
enum Comma {
    InText,
    EndsValue,
    OpensContainer,
}

/// The state of one string's scan.
struct StringScan {
    left: char,
    right: char,
    /// The string had no opening quote.
    missing: bool,
    /// The string opened with a doubled quote.
    doubled: bool,
    out: String,
    /// A quote inside the text is waiting for its pair.
    unmatched: bool,
    /// Inside a German low-high quotation („...”).
    low: bool,
    /// A comma announced a container that is part of the text.
    pending: bool,
    /// The open brackets of a container in the text.
    stack: Vec<char>,
}

impl StringScan {
    /// `c` joins the string's text.
    fn keep(&mut self, p: &mut Parser, c: char) -> ControlFlow<()> {
        self.out.push(c);
        p.index += 1;
        ControlFlow::Continue(())
    }
    fn track_nesting(&mut self, c: char) {
        if !matches!(c, '{' | '[') {
            return;
        }
        if self.pending {
            self.stack.push(c);
            self.pending = false;
        } else if !self.stack.is_empty() {
            self.stack.push(c);
        }
    }
}

/// A code fence, which may close a value written as a fenced block.
const FENCE: &str = "```";
const UNICODE_ESCAPE_DIGITS: usize = 4;
const HEX_ESCAPE_DIGITS: usize = 2;
const HIGH_SURROGATES: std::ops::RangeInclusive<u32> = 0xD800..=0xDBFF;
const LOW_SURROGATES: std::ops::RangeInclusive<u32> = 0xDC00..=0xDFFF;
/// The first code point past the Basic Multilingual Plane, where surrogate pairs start.
const SUPPLEMENTARY_PLANES: u32 = 0x10000;
/// The bits each surrogate half carries.
const SURROGATE_BITS: u32 = 10;

/// The character a UTF-16 surrogate pair encodes.
fn surrogate_pair(high: u32, low: u32) -> Option<char> {
    if !HIGH_SURROGATES.contains(&high) || !LOW_SURROGATES.contains(&low) {
        return None;
    }
    let offset =
        ((high - HIGH_SURROGATES.start()) << SURROGATE_BITS) + (low - LOW_SURROGATES.start());
    char::from_u32(SUPPLEMENTARY_PLANES + offset)
}

/// Whether the last `{` of `out` is unclosed, so a `}` closes it as text.
fn opens_literal_brace(out: &str) -> bool {
    let mut balance = 0;
    for x in out.chars().rev() {
        match x {
            '}' => balance += 1,
            '{' if balance == 0 => return true,
            '{' => balance -= 1,
            _ => {}
        }
    }
    false
}
impl Parser {
    pub(super) fn explicit_tuple(&self) -> bool {
        let mut i = 1;
        let mut stack = Vec::new();
        let mut content = false;
        while let Some(c) = self.at(i) {
            if quote(c) {
                if stack.is_empty() {
                    content = true;
                }
                i = self.skip(&[c], i + 1);
            } else if matches!(c, '(' | '[' | '{') {
                if stack.is_empty() {
                    content = true;
                }
                stack.push(c);
            } else if c == ')' && stack.is_empty() {
                return !content;
            } else if matches!(c, ')' | ']' | '}') {
                stack.pop();
            } else if c == ',' && stack.is_empty() {
                return true;
            } else if !c.is_whitespace() && stack.is_empty() {
                content = true;
            }
            i += 1;
        }
        !content
    }
    pub(super) fn top_paren(&self) -> bool {
        if !self.line_blank_before() {
            return false;
        }
        let first = self.at(self.scroll(1));
        if !first.is_some_and(|c| {
            quote(c)
                || c.is_ascii_digit()
                || matches!(c, ')' | '{' | '[' | '(' | '-' | '.' | 't' | 'f' | 'n')
        }) {
            return false;
        }
        let mut depth = 0;
        let mut i = 1;
        while let Some(c) = self.at(i) {
            if quote(c) {
                i = self.skip(&[c], i + 1);
            } else if c == '(' {
                depth += 1;
            } else if c == ')' {
                if depth == 0 {
                    return self.rest_of_line_blank(i + 1);
                }
                depth -= 1;
            }
            i += 1;
        }
        true
    }
    /// Only whitespace between the line's start and the current character.
    fn line_blank_before(&self) -> bool {
        let mut i = -1;
        while let Some(c) = self.at(i) {
            if matches!(c, '\n' | '\r') {
                return true;
            }
            if !c.is_whitespace() {
                return false;
            }
            i -= 1;
        }
        true
    }
    /// Only whitespace from offset `i` to the end of the line or of the input.
    fn rest_of_line_blank(&self, mut i: isize) -> bool {
        while let Some(c) = self.at(i) {
            if matches!(c, '\n' | '\r') {
                return true;
            }
            if !c.is_whitespace() {
                return false;
            }
            i += 1;
        }
        true
    }
    fn comment_scroll(&self, mut i: isize) -> isize {
        loop {
            i = self.scroll(i);
            match (self.at(i), self.at(i + 1)) {
                (Some('#'), _) | (Some('/'), Some('/')) => {
                    while self.at(i).is_some_and(|c| !matches!(c, '\n' | '\r')) {
                        i += 1;
                    }
                }
                (Some('/'), Some('*')) => {
                    i += 2;
                    while self.at(i).is_some()
                        && !(self.at(i) == Some('*') && self.at(i + 1) == Some('/'))
                    {
                        i += 1;
                    }
                    if self.at(i).is_some() {
                        i += 2;
                    }
                }
                _ => return i,
            }
        }
    }
    fn inline_end(&self, mut i: isize) -> Option<isize> {
        let mut stack = vec![match self.at(i)? {
            '[' => ']',
            '{' => '}',
            '(' => ')',
            _ => return Some(i),
        }];
        i += 1;
        while let Some(c) = self.at(i) {
            if quote(c) {
                i = self.skip(&[closing(c)], i + 1);
                self.at(i)?;
            } else if matches!(c, '[' | '{' | '(') {
                stack.push(match c {
                    '[' => ']',
                    '{' => '}',
                    _ => ')',
                });
            } else if stack.last() == Some(&c) {
                stack.pop();
                if stack.is_empty() {
                    return Some(i + 1);
                }
            }
            i += 1;
        }
        None
    }
}
