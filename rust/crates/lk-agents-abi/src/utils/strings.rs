//! Small pure string helpers shared with JS.

/// `CamelCase` -> `snake_case`, matching the reference's two-regex form exactly:
/// `([A-Z]+)([A-Z][a-z])` then `([a-z0-9])([A-Z])`, lowercased.
#[crate::host(export(pid = false))]
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

/// Is this a LiveKit Cloud URL? Hostname suffix only, so a path or query cannot fake it.
#[crate::host(export(pid = false, name = "is_cloud"))]
pub fn is_cloud_url(url: &str) -> bool {
    let Some(host) = host_of(url) else { return false };
    host.ends_with(".livekit.cloud") || host.ends_with(".livekit.run")
}

/// Minimal `urlparse().hostname`: scheme, optional userinfo, strip port and IPv6 brackets, lowercase.
pub(crate) fn host_of(url: &str) -> Option<String> {
    let rest = url.split_once("://").map(|(_, r)| r)?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let authority = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    if authority.is_empty() {
        return None;
    }
    let host = if let Some(end) = authority.strip_prefix('[').and_then(|a| a.find(']')) {
        &authority[1..=end]
    } else {
        authority.split(':').next()?
    };
    (!host.is_empty()).then(|| host.to_ascii_lowercase())
}

#[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
/// `utils.shortuuid(prefix)`: `prefix + uuid.uuid4().hex[:12]`, twelve hex characters, all before
/// the version nibble, so six random bytes are equivalent.
pub fn shortuuid(prefix: &str) -> String {
    use rand::RngCore;
    use std::fmt::Write as _;
    let mut bytes = [0u8; 6];
    rand::rng().fill_bytes(&mut bytes);
    let mut out = String::with_capacity(prefix.len() + 12);
    out.push_str(prefix);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
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

#[cfg(test)]
mod tests {
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

    // Catches: matching the suffix against the whole URL, an undotted suffix, or the port or
    // userinfo leaking into the hostname.
    #[test]
    fn only_the_hostname_decides_cloud() {
        for (url, cloud) in [
            ("wss://my-project.livekit.cloud", true),
            ("https://x.livekit.run/rtc?token=1", true),
            ("wss://p.livekit.cloud:443/x", true),
            ("wss://user:pw@p.livekit.cloud", true),
            ("https://evil.com/?x=.livekit.cloud", false),
            ("https://notlivekit.cloud", false),
            ("not a url", false),
            ("wss://[::1]:7880", false),
        ] {
            assert_eq!(is_cloud_url(url), cloud, "{url}");
        }
    }
}
