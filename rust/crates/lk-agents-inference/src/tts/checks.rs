//! The checks a host synthesis's end applies (`ChunkedStream` and `SynthesizeStream`).

use lk_agents_abi::utils::strings::PYTHON_WHITESPACE;

pub struct SynthesisChecks;

impl lk_agents_abi::tts::streams::SynthesisChecks for SynthesisChecks {
    fn validate_synthesis(
        text: &str,
        duration: f64,
        segments: Option<(usize, usize)>,
    ) -> Option<String> {
        if text.trim_matches(|c| PYTHON_WHITESPACE.contains(&c)).is_empty() {
            return None;
        }
        if duration <= 0.0 {
            return Some(format!("no audio frames were pushed for text: {text}"));
        }
        match segments {
            Some((expected, actual)) if expected != actual => Some(format!(
                "number of segments mismatch: expected {expected}, but got {actual}"
            )),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use lk_agents_abi::tts::streams::SynthesisChecks as _;

    use super::SynthesisChecks;

    #[test]
    fn synthesis_without_audio_or_with_lost_segments_fails() {
        assert!(SynthesisChecks::validate_synthesis("hello", 0.0, None).is_some());
        assert!(SynthesisChecks::validate_synthesis("hello", 1.0, Some((2, 1))).is_some());
    }
}
