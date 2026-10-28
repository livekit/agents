//! Character classification shared by text processing and speech tokenization.

/// Python's whitespace set includes the four ASCII information separators.
pub fn is_py_space(c: char) -> bool {
    lk_agents_abi::utils::strings::PYTHON_WHITESPACE.contains(&c)
}

/// Trim with Python's whitespace set.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// `CamelCase` -> `snake_case`, matching the reference's two-regex form exactly:
/// `([A-Z]+)([A-Z][a-z])` then `([a-z0-9])([A-Z])`, lowercased.
pub fn camel_to_snake_case(name: &str) -> String {
    let c: Vec<char> = name.chars().collect();
    let mut out = String::with_capacity(name.len() + 4);

    for i in 0..c.len() {
        // Boundary 1: an acronym run ending (`HTTPServer` -> `HTTP_Server`).
        let acronym_end = i > 0
            && c[i].is_ascii_uppercase()
            && c[i - 1].is_ascii_uppercase()
            && c.get(i + 1).is_some_and(char::is_ascii_lowercase);
        // Boundary 2: lower/digit followed by upper, e.g. `parseURL` -> `parse_URL`.
        let word_start = i > 0
            && c[i].is_ascii_uppercase()
            && (c[i - 1].is_ascii_lowercase() || c[i - 1].is_ascii_digit());

        if acronym_end || word_start {
            out.push('_');
        }
        out.extend(c[i].to_lowercase());
    }
    out
}

/// Python's `repr()` of a string, so multi-line text stays on one line. Unlike CPython,
/// unprintable non-ASCII is not escaped.
pub fn py_repr(s: &str) -> String {
    // CPython picks `'` unless that would need escaping and `"` would not.
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

pub struct TextAlgorithms;
impl lk_agents_abi::utils::strings::TextFunctions for TextAlgorithms {
    fn camel_to_snake_case(name: &str) -> String {
        camel_to_snake_case(name)
    }
}

#[cfg(test)]
mod string_tests {
    use super::*;
    // Catches: collapsing the two regexes into one (breaks acronyms), digits treated as capitals,
    // or the acronym boundary testing the current character (`http__server`).
    #[test]
    fn camel_case_converts_the_way_the_reference_regexes_do() {
        for (camel, snake) in [
            ("OpenAILLM", "open_aillm"),
            ("HTTPServer", "http_server"),
            ("parseURL", "parse_url"),
            ("Sha256Hash", "sha256_hash"),
            ("A1B2", "a1_b2"),
            ("already_snake", "already_snake"),
            ("", ""),
            ("A", "a"),
            ("HTTP_Server", "http_server"),
            ("Open_AI", "open_ai"),
            ("a_Bc", "a_bc"),
        ] {
            assert_eq!(camel_to_snake_case(camel), snake, "{camel}");
        }
    }
}
