//! Character classification shared by text processing and speech tokenization.

/// Python's whitespace set includes the four ASCII information separators.
pub fn is_py_space(c: char) -> bool {
    lk_agents_abi::utils::strings::PYTHON_WHITESPACE.contains(&c)
}

/// Trim with Python's whitespace set.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}
