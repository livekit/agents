//! Character classification shared by text processing and speech tokenization.

/// Python's whitespace set includes the four ASCII information separators.
pub fn is_py_space(c: char) -> bool {
    lk_agents_abi::utils::strings::PYTHON_WHITESPACE.contains(&c)
}

/// Trim with Python's whitespace set.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
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
