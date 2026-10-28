//! Text classification data and host utility declarations.

/// Characters accepted by Python's whitespace classification, including ASCII separators.
pub const PYTHON_WHITESPACE: &[char] = &[
    '\t', '\n', '\u{b}', '\u{c}', '\r', '\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', ' ', '\u{85}',
    '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}',
    '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}',
    '\u{205f}', '\u{3000}',
];

#[crate::host(functions(pid = false))]
pub trait TextFunctions {
    #[host(function)]
    fn camel_to_snake_case(name: &str) -> String;
}

#[crate::host(functions(pid = false))]
pub trait TransportFunctions {
    #[host(function, name = "is_cloud")]
    fn is_cloud_url(url: &str) -> bool;
    #[host(function, pid = true)]
    fn observability_url(livekit_url: &str, env_override: Option<String>) -> Option<String>;
}
