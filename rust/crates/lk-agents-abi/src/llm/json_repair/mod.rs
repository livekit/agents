//! Native translation of json_repair v0.60.1's schema-free, non-streaming recovery path.
//! Copyright (c) 2023 Stefano Baccianella. MIT; see tests/fixtures/json_repair/LICENSE.
use serde_json::{Map, Value};
mod string;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Context {
    Key,
    Value,
    Array,
}
struct Parser {
    chars: Vec<char>,
    index: usize,
    contexts: Vec<Context>,
    depth: usize,
    limit_hit: bool,
}
fn empty() -> Value {
    Value::String(String::new())
}
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(v) => *v,
        Value::String(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
        Value::Number(v) => v.as_f64() != Some(0.0),
    }
}
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|b| same(v, b)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(a, b))
        }
        (Value::Null, Value::Null)
        | (Value::Bool(_), Value::Bool(_))
        | (Value::String(_), Value::String(_)) => true,
        (Value::Number(a), Value::Number(b)) => a.is_f64() == b.is_f64(),
        _ => false,
    }
}
fn quote(c: char) -> bool {
    matches!(c, '"' | '\'' | '“' | '”')
}
fn closing(c: char) -> char {
    if c == '“' { '”' } else { c }
}
impl Parser {
    fn at(&self, offset: isize) -> Option<char> {
        self.index.checked_add_signed(offset).and_then(|i| self.chars.get(i)).copied()
    }
    fn current(&self) -> Option<Context> {
        self.contexts.last().copied()
    }
    fn has(&self, c: Context) -> bool {
        self.contexts.contains(&c)
    }
    fn scroll(&self, mut i: isize) -> isize {
        while self.at(i).is_some_and(char::is_whitespace) {
            i += 1;
        }
        i
    }
    fn ws(&mut self) {
        self.index += self.scroll(0) as usize;
    }
    fn skip(&self, targets: &[char], mut i: isize) -> isize {
        let mut slashes = 0;
        while let Some(c) = self.at(i) {
            if c == '\\' {
                slashes += 1;
            } else {
                if targets.contains(&c) && slashes % 2 == 0 {
                    break;
                }
                slashes = 0;
            }
            i += 1;
        }
        i
    }
    fn slice(&self, a: usize, b: usize) -> String {
        self.chars[a.min(self.chars.len())..b.min(self.chars.len())].iter().collect()
    }
    fn in_context(&mut self, c: Context, f: impl FnOnce(&mut Self) -> Value) -> Value {
        self.contexts.push(c);
        let v = f(self);
        self.contexts.pop();
        v
    }
    fn json(&mut self) -> Value {
        if self.depth >= 64 {
            self.limit_hit = true;
            self.index = self.chars.len();
            return empty();
        }
        self.depth += 1;
        let value = loop {
            match self.at(0) {
                None => break empty(),
                Some('{') => {
                    self.index += 1;
                    break self.object();
                }
                Some('[') => {
                    self.index += 1;
                    break self.array(']');
                }
                Some('(') => {
                    if self.current().is_none() && !self.top_paren() {
                        self.index += 1;
                        continue;
                    }
                    let tuple = self.explicit_tuple();
                    self.index += 1;
                    let mut arr = self.array(')');
                    if !tuple {
                        if let Value::Array(ref mut a) = arr {
                            if a.len() == 1 {
                                arr = a.pop().unwrap();
                            }
                        }
                    }
                    break arr;
                }
                Some(c) if self.current().is_some() && (quote(c) || c.is_alphabetic()) => {
                    break self.string();
                }
                Some(c)
                    if self.current().is_some() && (c.is_ascii_digit() || c == '-' || c == '.') =>
                {
                    break self.number();
                }
                Some('#' | '/') => break self.comment(),
                _ => self.index += 1,
            }
        };
        self.depth -= 1;
        value
    }
    fn comment(&mut self) -> Value {
        loop {
            match (self.at(0), self.at(1)) {
                (Some('#'), _) => {
                    while self.at(0).is_some_and(|c| {
                        !matches!(c, '\n' | '\r')
                            && !(c == ']' && self.has(Context::Array))
                            && !(c == '}' && self.has(Context::Value))
                    }) {
                        self.index += 1;
                    }
                }
                (Some('/'), Some('/')) => {
                    self.index += 2;
                    while self.at(0).is_some_and(|c| !matches!(c, '\n' | '\r')) {
                        self.index += 1;
                    }
                }
                (Some('/'), Some('*')) => {
                    self.index += 2;
                    while self.at(0).is_some() {
                        let done = self.at(0) == Some('*') && self.at(1) == Some('/');
                        self.index += 1;
                        if done {
                            self.index += 1;
                            break;
                        }
                    }
                }
                _ => self.index += 1,
            }
            if self.current().is_some() {
                return empty();
            }
            self.ws();
            if !matches!(self.at(0), Some('#' | '/')) {
                return self.json();
            }
        }
    }
    fn number(&mut self) -> Value {
        let start = self.index;
        let mut number = String::new();
        while let Some(c) = self.at(0) {
            if !"0123456789-.eE/,_".contains(c)
                || (c == ',' && self.current() == Some(Context::Array))
            {
                break;
            }
            if c != '_' {
                number.push(c);
            }
            self.index += 1;
        }
        if self.at(0).is_some_and(char::is_alphabetic) {
            self.index = start;
            return self.string();
        }
        if number.ends_with(['-', 'e', 'E', '/', ',']) {
            number.pop();
            self.index = self.index.saturating_sub(1);
        }
        if number.contains(',') {
            return Value::String(number);
        }
        if let Ok(n) = number.parse::<i64>() {
            return n.into();
        }
        if let Ok(n) = number.parse::<u64>() {
            return n.into();
        }
        if let Ok(n) = number.parse::<f64>() {
            if let Some(n) = serde_json::Number::from_f64(n) {
                return n.into();
            }
        }
        Value::String(number)
    }
    fn array(&mut self, end: char) -> Value {
        if self.depth >= 64 {
            self.limit_hit = true;
            self.index = self.chars.len();
            return empty();
        }
        self.depth += 1;
        let result = self.array_body(end);
        self.depth -= 1;
        result
    }
    fn array_body(&mut self, end: char) -> Value {
        self.in_context(Context::Array, |p| {
            let mut arr = Vec::new();
            p.ws();
            while p.at(0).is_some_and(|c| c != end && c != '}') {
                let before = p.index;
                let value = if p.at(0).is_some_and(quote) {
                    let q = p.at(0).unwrap();
                    let i = p.skip(&[q], 1);
                    let i = p.scroll(i + 1);
                    if p.at(i) == Some(':') { p.object() } else { p.string() }
                } else {
                    p.json()
                };
                if matches!(&value,Value::String(s) if s.is_empty())
                    || matches!(&value,Value::Array(a) if a.is_empty())
                    || matches!(&value,Value::Object(o) if o.is_empty())
                {
                    if !matches!(p.at(0),Some(c) if c==end||c==',') {
                        p.index += 1;
                    } else {
                        arr.push(value);
                    }
                } else if !(value == Value::String("...".into()) && p.at(-1) == Some('.')) {
                    arr.push(value);
                }
                while p.at(0).is_some_and(|c| c != end && (c.is_whitespace() || c == ',')) {
                    p.index += 1;
                }
                if p.index == before {
                    p.index += 1;
                }
            }
            p.index += 1;
            Value::Array(arr)
        })
    }
    fn object(&mut self) -> Value {
        if self.depth >= 64 {
            self.limit_hit = true;
            self.index = self.chars.len();
            return empty();
        }
        self.depth += 1;
        let result = self.object_body();
        self.depth -= 1;
        result
    }
    fn object_body(&mut self) -> Value {
        let mut obj = Map::new();
        let start = self.index;
        let parent = self.current();
        while self.at(0).is_some_and(|c| c != '}') {
            let before = self.index;
            self.ws();
            if self.at(0) == Some(':') {
                self.index += 1;
            }
            let mut rollback = self.index;
            let key = self
                .in_context(Context::Key, |p| {
                    loop {
                        rollback = p.index;
                        if p.at(0) == Some('[') && !obj.is_empty() {
                            if let Some(Value::Array(prev)) = obj.values_mut().last() {
                                p.index += 1;
                                if let Value::Array(next) = p.array(']') {
                                    let lengths: Vec<usize> = prev
                                        .iter()
                                        .filter_map(|v| v.as_array().map(Vec::len))
                                        .collect();
                                    let width = lengths
                                        .first()
                                        .copied()
                                        .filter(|n| *n > 0 && lengths.iter().all(|x| x == n));
                                    if let Some(width) = width {
                                        let tail_start = prev
                                            .iter()
                                            .rposition(Value::is_array)
                                            .map(|i| i + 1)
                                            .unwrap_or(0);
                                        let tail = prev.split_off(tail_start);
                                        if tail.len() % width == 0 {
                                            prev.extend(
                                                tail.chunks(width)
                                                    .map(|c| Value::Array(c.to_vec())),
                                            );
                                        } else {
                                            prev.extend(tail);
                                        }
                                        if !next.is_empty() {
                                            if next.iter().all(Value::is_array) {
                                                prev.extend(next);
                                            } else {
                                                prev.push(Value::Array(next));
                                            }
                                        }
                                    } else {
                                        if next.len() == 1 && next[0].is_array() {
                                            prev.extend(next[0].as_array().unwrap().clone());
                                        } else {
                                            prev.extend(next);
                                        }
                                    }
                                }
                                p.ws();
                                if p.at(0) == Some(',') {
                                    p.index += 1;
                                }
                                p.ws();
                                continue;
                            }
                        }
                        let key = p.string();
                        if key == empty() {
                            p.ws();
                        }
                        if key != empty() || matches!(p.at(0), Some(':' | '}')) || p.at(0).is_none()
                        {
                            break key;
                        }
                        p.ws();
                    }
                })
                .as_str()
                .unwrap_or("")
                .to_owned();
            if self.has(Context::Array) && parent != Some(Context::Value) && obj.contains_key(&key)
            {
                let mut prev = rollback as isize - self.index as isize - 1;
                while self.at(prev).is_some_and(char::is_whitespace) {
                    prev -= 1;
                }
                if !(self.chars.get(rollback).copied().is_some_and(quote)
                    && self.at(prev) == Some(',')
                    && self.at(self.scroll(0)) == Some(':'))
                {
                    self.chars.insert(rollback, '{');
                    self.index = rollback.saturating_sub(1);
                    break;
                }
            }
            self.ws();
            if self.at(0).is_none() || self.at(0) == Some('}') {
                continue;
            }
            self.index += 1;
            let value = self.in_context(Context::Value, |p| {
                p.ws();
                if matches!(p.at(0), Some(',' | '}')) { empty() } else { p.json() }
            });
            obj.insert(key, value);
            if matches!(self.at(0), Some(',' | '\'' | '"')) {
                self.index += 1;
            }
            if self.at(0) == Some(']') && self.has(Context::Array) {
                self.index = self.index.saturating_sub(1);
                break;
            }
            self.ws();
            if before == self.index {
                self.index += 1;
            }
        }
        self.index += 1;
        if obj.is_empty() && self.index.saturating_sub(start) > 2 {
            let body = self.slice(start, self.index.saturating_sub(1));
            if body.trim_start().starts_with("\\\"") {
                let normalized = self
                    .slice(start.saturating_sub(1), self.index)
                    .replace("\\\"", "\"")
                    .replace("\\'", "'");
                return match repair(&normalized) {
                    Ok(v) => v,
                    Err(_) => {
                        self.limit_hit = true;
                        empty()
                    }
                };
            }

            if !body.contains(':')
                && !body.trim().is_empty()
                && !body.trim().starts_with(['#', '/'])
            {
                self.index = start;
                let result = self.in_context(Context::Key, |p| p.array(']'));
                self.contexts.push(Context::Key);
                return result;
            }
        }
        if let Some(context) = self.current() {
            if self.at(0) == Some('}') && !matches!(context, Context::Key | Context::Value) {
                self.index += 1;
            }
        } else {
            self.ws();
            if self.at(0) == Some(',') {
                self.index += 1;
                self.ws();
                if self.at(0).is_some_and(quote) {
                    if let Value::Object(add) = self.object() {
                        obj.extend(add);
                    }
                }
            }
        }
        Value::Object(obj)
    }
    fn parse(&mut self) -> Value {
        let first = self.json();
        if self.index >= self.chars.len() {
            return first;
        }
        let mut values = vec![first];
        while self.index < self.chars.len() {
            self.contexts.clear();
            let mut back = -1;
            while self.at(back).is_some_and(char::is_whitespace) {
                back -= 1;
            }
            let comma = self.at(self.scroll(0)) == Some(',') || self.at(back) == Some(',');
            let next = self.json();
            if truthy(&next) {
                if !comma && same(values.last().unwrap(), &next) || !truthy(values.last().unwrap())
                {
                    values.pop();
                }
                values.push(next);
            } else {
                self.index += 1;
            }
        }
        if values.len() == 1 { values.pop().unwrap() } else { Value::Array(values) }
    }
}
/// JSON-compatible recovery. No file handles, language runtimes, schema callbacks or logging sinks.
pub fn repair(raw: &str) -> Result<Value, String> {
    if let Ok(value) = serde_json::from_str(raw) {
        return Ok(value);
    }
    let mut parser = Parser {
        chars: raw.chars().collect(),
        index: 0,
        contexts: Vec::new(),
        depth: 0,
        limit_hit: false,
    };
    let value = parser.parse();
    if parser.limit_hit {
        Err("JSON repair exceeded maximum nesting depth".into())
    } else {
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_inputs_terminate_without_panicking() {
        let alphabet: Vec<char> = "{}[]():,\"'`/\\ntf01 abc\n“”".chars().collect();
        let mut seed = 42u64;
        for _ in 0..2000 {
            let input: String = (0..80)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    alphabet[(seed as usize >> 16) % alphabet.len()]
                })
                .collect();
            let _ = repair(&input);
        }
        assert!(repair(&format!("{{'x':{}", "[".repeat(4096))).is_err());
    }
    #[test]
    fn upstream_corpus() {
        let cases: Vec<Value> =
            serde_json::from_str(include_str!("../../../tests/fixtures/json_repair/cases.json"))
                .unwrap();
        let mut failures = Vec::new();
        for (i, case) in cases.iter().enumerate() {
            let got = repair(case["input"].as_str().unwrap()).unwrap();
            if got != case["expected"] {
                failures.push(serde_json::json!({"index":i,"input":case["input"],"expected":case["expected"],"got":got}));
            }
        }
        assert!(
            failures.is_empty(),
            "{} / {} upstream cases failed: {}",
            failures.len(),
            cases.len(),
            serde_json::to_string_pretty(&failures).unwrap()
        );
    }
}
