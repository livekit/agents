//! Provider attempt records: the request kind and the decision a host enacts. The policy is
//! `lk_agents_net::attempts`.

/// Crosses to a host as its tag (`FromStr` / `as_str`).
/// The operation whose failures share an attempt allowance.
#[crate::host(string)]
#[crate::host(wire_strs(err = UnknownRequestKind, msg = UnknownRequestKind(s.to_owned())))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestKind {
    Llm,
    SttBatch,
    SttStream,
    TtsBatch,
    TtsStream,
    Interruption,
    Avatar,
}

/// `Display` does not interpolate the value: the binding has always raised the bare message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownRequestKind(pub String);

impl std::fmt::Display for UnknownRequestKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("unknown provider request kind")
    }
}

impl std::error::Error for UnknownRequestKind {}

#[derive(Debug, PartialEq)]
#[crate::host(record(module = "_retry", name = "_core_RetryDecision", emit))]
pub enum RetryDecision {
    Stop,
    Raise,
    #[host(field = "message")]
    Exhausted(String),
    #[host(field = "delay")]
    Retry(f64),
}

#[crate::host(export(pid = false))]
pub fn validate_synthesis(
    text: &str,
    duration: f64,
    #[host(required)] segments: Option<(usize, usize)>,
) -> Option<String> {
    if text.trim_matches(|c| crate::utils::strings::PYTHON_WHITESPACE.contains(&c)).is_empty() {
        return None;
    }
    if duration <= 0.0 {
        return Some(format!("no audio frames were pushed for text: {text}"));
    }
    if let Some((expected, actual)) = segments {
        if expected != actual {
            return Some(format!(
                "number of segments mismatch: expected {expected}, but got {actual}"
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synthesis_without_audio_or_with_lost_segments_fails() {
        assert!(validate_synthesis("hello", 0.0, None).is_some());
        assert!(validate_synthesis("hello", 1.0, Some((2, 1))).is_some());
    }
}
