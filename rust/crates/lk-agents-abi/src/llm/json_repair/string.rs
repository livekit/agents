//! Translation of parse_string.py and parse_string_helpers from the attributed upstream.
use super::*;
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
    fn comma_kind(&self) -> u8 {
        let i = self.scroll(1);
        match self.at(i) {
            None | Some('}') => return 1,
            Some('{' | '[') => return 2,
            _ => {}
        }
        if self.at(i).is_some_and(quote) {
            let q = self.at(i).unwrap();
            let end = self.skip(&[q], i + 1);
            return if self.at(end).is_some() && self.at(self.scroll(end + 1)) == Some(':') {
                1
            } else {
                0
            };
        }
        if self.member(if self.at(i) == Some('`') { i + 1 } else { i }) {
            return 1;
        }
        let q = self.skip(&['"', '\'', '“', '”'], i);
        let c = self.skip(&['{', '['], i);
        if self.at(c).is_some() && c < q {
            return 0;
        }
        if self.at(i).is_some_and(quote) {
            return 0;
        }
        if self.member(q) { 1 } else { 0 }
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
        let mut out = String::new();
        let mut unmatched = false;
        let mut low = false;
        let mut pending = false;
        let mut stack = Vec::new();
        while let Some(c) = self.at(0) {
            if c == right && !low {
                break;
            }
            if missing
                && ((self.current() == Some(Context::Key) && (c == ':' || c.is_whitespace()))
                    || (self.current() == Some(Context::Array) && matches!(c, ']' | ',')))
            {
                break;
            }
            if c == '„' && !out.ends_with('\\') {
                low = true;
                out.push(c);
                self.index += 1;
                continue;
            }
            if low && c == '”' {
                low = false;
                out.push(c);
                self.index += 1;
                continue;
            }
            if pending && matches!(c, '[' | '{' | '(') {
                if let Some(end) = self.inline_end(0) {
                    out.push_str(&self.slice(self.index, self.index + end as usize));
                    self.index += end as usize;
                    pending = false;
                    stack.clear();
                    continue;
                }
            }
            if self.current() == Some(Context::Value) && c == ',' && !pending && stack.is_empty() {
                match self.comma_kind() {
                    1 => break,
                    2 => pending = true,
                    _ => {}
                }
                out.push(c);
                self.index += 1;
                continue;
            }
            if matches!(c, '{' | '[') {
                if pending {
                    stack.push(c);
                    pending = false;
                } else if !stack.is_empty() {
                    stack.push(c);
                }
            }
            if matches!((c, stack.last()), ('}', Some('{')) | (']', Some('['))) {
                stack.pop();
                out.push(c);
                self.index += 1;
                continue;
            }
            if self.current() == Some(Context::Value) && c == '}' && !out.ends_with(right) {
                let mut balance = 0;
                let mut literal = false;
                for x in out.chars().rev() {
                    if x == '}' {
                        balance += 1;
                    } else if x == '{' {
                        if balance == 0 {
                            literal = true;
                            break;
                        }
                        balance -= 1;
                    }
                }
                if literal {
                    out.push(c);
                    self.index += 1;
                    continue;
                }
                let mut missing_right = true;
                if self.at(1) == Some('\\') {
                    missing_right = false;
                }
                let mut i = self.skip(&[right], 1);
                if self.at(i).is_some() {
                    i = self.scroll(i + 1);
                    if matches!(self.at(i), None | Some(',' | '}')) {
                        missing_right = false;
                    } else {
                        i = self.skip(&[left], i);
                        if self.at(i).is_none() {
                            missing_right = false;
                        } else {
                            i = self.scroll(i + 1);
                            if self.at(i).is_some_and(|c| c != ':') {
                                missing_right = false;
                            }
                        }
                    }
                } else {
                    if self.at(self.skip(&[':'], 1)).is_some() {
                        break;
                    }
                    i = self.scroll(1);
                    let j = self.skip(&['}'], i);
                    if j - i > 1 {
                        missing_right = false;
                    }
                }
                if missing_right {
                    break;
                }
            }
            if c == ']'
                && self.has(Context::Array)
                && !out.ends_with(right)
                && self.at(self.skip(&[right], 0)).is_none()
            {
                break;
            }
            if self.current() == Some(Context::Value) && c == '}' {
                let i = self.scroll(1);
                if self.at(i).is_none() {
                    break;
                }
                if self.at(i) == Some('`')
                    && self.at(i + 1) == Some('`')
                    && self.at(i + 2) == Some('`')
                {
                    let mut start = self.comment_scroll(i + 3);
                    let mut keep_container = false;
                    if self.at(start).is_some_and(|c| matches!(c, '[' | '{' | '(')) {
                        if let Some(end) = self.inline_end(start) {
                            let after = self.scroll(end);
                            let ch = self.at(after);
                            if matches!(ch, None | Some('}'))
                                || ch == Some(',')
                                    && (self
                                        .at(self.comment_scroll(after + 1))
                                        .is_none_or(|c| c == '}')
                                        || self.member(self.comment_scroll(after + 1)))
                            {
                                break;
                            }
                            keep_container = true;
                            start = end;
                        }
                    }
                    let mut q = self.skip(&[right], start);
                    while self.at(q).is_some() {
                        if matches!(self.at(self.scroll(q + 1)), None | Some(',' | '}' | ']')) {
                            break;
                        }
                        q = self.skip(&[right], q + 1);
                    }
                    if self.at(q).is_none() {
                        break;
                    }
                    if keep_container {
                        pending = true;
                    }
                }
            }
            out.push(c);
            self.index += 1;
            if out.ends_with('\\') {
                if let Some(e) = self.at(0) {
                    if matches!(e, 't' | 'n' | 'r' | 'b' | '\\')
                        || e == right
                        || quote(e)
                        || e == '„'
                    {
                        out.pop();
                        out.push(match e {
                            't' => '\t',
                            'n' => '\n',
                            'r' => '\r',
                            'b' => '\u{8}',
                            _ => e,
                        });
                        self.index += 1;
                        while out.ends_with('\\')
                            && self.at(0).is_some_and(|c| c == right || c == '\\')
                        {
                            out.pop();
                            out.push(self.at(0).unwrap());
                            self.index += 1;
                        }
                        continue;
                    }
                    if e == 'u' || e == 'x' {
                        let n = if e == 'u' { 4 } else { 2 };
                        let code = self.slice(self.index + 1, self.index + 1 + n);
                        if code.len() == n && code.chars().all(|c| c.is_ascii_hexdigit()) {
                            let u = u32::from_str_radix(&code, 16).unwrap();
                            if (0xD800..=0xDBFF).contains(&u)
                                && self.at((n + 1) as isize) == Some('\\')
                                && self.at((n + 2) as isize) == Some('u')
                            {
                                let tail = self.slice(self.index + n + 3, self.index + n + 7);
                                if let Ok(low) = u32::from_str_radix(&tail, 16) {
                                    if (0xDC00..=0xDFFF).contains(&low) {
                                        out.pop();
                                        out.push(
                                            char::from_u32(
                                                0x10000 + ((u - 0xD800) << 10) + (low - 0xDC00),
                                            )
                                            .unwrap(),
                                        );
                                        self.index += n + 7;
                                        continue;
                                    }
                                }
                            }

                            if let Some(ch) = char::from_u32(u) {
                                out.pop();
                                out.push(ch);
                                self.index += 1 + n;
                                continue;
                            }
                        }
                    }
                }
            }
            if self.at(0) == Some(':') && !missing && self.current() == Some(Context::Key) {
                let i = self.skip(&[left], 1);
                if self.at(i).is_none() {
                    break;
                }
                let i = self.skip(&[right], i + 1);
                if self.at(i).is_some() && matches!(self.at(self.scroll(i + 1)), Some(',' | '}')) {
                    break;
                }
            }
            if low && self.at(0) == Some('"') {
                low = false;
                out.push('"');
                self.index += 1;
                continue;
            }
            if self.at(0) == Some(right) && !out.ends_with('\\') {
                if doubled && self.at(1) == Some(right) {
                    self.index += 1;
                    continue;
                }
                if missing && self.current() == Some(Context::Value) {
                    let i = self.skip(&[right, left], 1);
                    if self.at(i).is_some() && self.at(self.scroll(i + 1)) == Some(':') {
                        self.index = self.index.saturating_sub(1);
                        break;
                    }
                    continue;
                }
                if unmatched {
                    unmatched = false;
                    out.push(right);
                    self.index += 1;
                    continue;
                }
                let mut i = 1;
                let mut check_comma = true;
                while self.at(i).is_some_and(|c| c != right && c != left) {
                    let ch = self.at(i).unwrap();
                    if check_comma && ch.is_alphabetic() {
                        check_comma = false;
                    }
                    if self.has(Context::Key) && matches!(ch, ':' | '}')
                        || self.has(Context::Value) && ch == '}'
                        || self.has(Context::Array) && matches!(ch, ']' | ',')
                        || check_comma && self.current() == Some(Context::Value) && ch == ','
                    {
                        break;
                    }
                    i += 1;
                }
                let next = self.at(i);
                if next == Some(',') && self.current() == Some(Context::Value) {
                    i = self.skip(&[right], i + 1);
                    if matches!(self.at(self.scroll(i + 1)), Some('}' | ',')) {
                        out.push(right);
                        self.index += 1;
                        continue;
                    }
                } else if next == Some(right) && self.at(i - 1) != Some('\\') {
                    let follows = |p: &Self, j: isize| {
                        let c = p.scroll(j + 1);
                        p.at(c) == Some(',') && p.member(p.comment_scroll(c + 1))
                    };
                    if (1..i).all(|j| self.at(j).is_none_or(char::is_whitespace))
                        && !(self.current() == Some(Context::Value) && follows(self, i))
                    {
                        break;
                    }
                    if self.current() == Some(Context::Value) {
                        if follows(self, i) {
                            out.push(right);
                            self.index += 1;
                            continue;
                        }
                        i = self.skip(&[right], i + 1) + 1;
                        while self.at(i).is_some_and(|c| c != ':') {
                            let ch = self.at(i).unwrap();
                            if matches!(ch, ',' | ']' | '}')
                                || (ch == right && self.at(i - 1) != Some('\\'))
                            {
                                break;
                            }
                            i += 1;
                        }
                        if self.at(i) != Some(':') {
                            unmatched = !unmatched;
                            out.push(right);
                            self.index += 1;
                            continue;
                        }
                    } else if self.current() == Some(Context::Array) {
                        let mut even = true;
                        while self.at(i) == Some(right) {
                            i = self.skip(&[right, ']'], i + 1);
                            if self.at(i) != Some(right) {
                                even = false;
                                break;
                            }
                            i = self.skip(&[right, ']'], i + 1);
                        }
                        if even {
                            unmatched = !unmatched;
                            out.push(right);
                            self.index += 1;
                            continue;
                        }
                        break;
                    } else if self.current() == Some(Context::Key) {
                        out.push(right);
                        self.index += 1;
                        continue;
                    }
                }
            }
        }
        if missing
            && self.current() == Some(Context::Key)
            && self.at(0).is_some_and(char::is_whitespace)
        {
            self.ws();
            if !matches!(self.at(0), Some(':' | ',')) {
                return empty();
            }
        }
        if self.at(0) != Some(right) {
            out = out.trim_end().to_owned();
        } else {
            self.index += 1;
        }
        if missing || out.ends_with('\n') {
            out = out.trim_end().to_owned();
        }
        Value::String(out)
    }
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
        let mut i = -1;
        while let Some(c) = self.at(i) {
            if matches!(c, '\n' | '\r') {
                break;
            }
            if !c.is_whitespace() {
                return false;
            }
            i -= 1;
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
        i = 1;
        while let Some(c) = self.at(i) {
            if quote(c) {
                i = self.skip(&[c], i + 1);
            } else if c == '(' {
                depth += 1;
            } else if c == ')' {
                if depth == 0 {
                    i += 1;
                    while let Some(t) = self.at(i) {
                        if matches!(t, '\n' | '\r') {
                            return true;
                        }
                        if !t.is_whitespace() {
                            return false;
                        }
                        i += 1;
                    }
                    return true;
                }
                depth -= 1;
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
