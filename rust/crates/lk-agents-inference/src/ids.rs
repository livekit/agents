//! `utils.misc.shortuuid` and `inference/_utils.create_inference_request_id`.

use rand::RngCore;

/// `prefix + uuid.uuid4().hex[:12]`: twelve hex characters, all before the version nibble, so six
/// random bytes are equivalent.
pub fn shortuuid(prefix: &str) -> String {
    let mut bytes = [0u8; 6];
    rand::rng().fill_bytes(&mut bytes);
    let mut out = String::with_capacity(prefix.len() + 12);
    out.push_str(prefix);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Which kind of logical request an id is minted for (`Literal["tts", "eot"]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    Tts,
    Eot,
}

impl RequestKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RequestKind::Tts => "tts",
            RequestKind::Eot => "eot",
        }
    }
}

/// A logical request ID prefixed with its connection's session id; `fallback_prefix` when there
/// is none.
pub fn create_inference_request_id(
    session_id: Option<&str>,
    kind: RequestKind,
    fallback_prefix: &str,
) -> String {
    // R2b: `if session_id` -- an empty session id takes the fallback branch.
    let prefix = match session_id {
        Some(sid) if !sid.is_empty() => format!("{sid}_{}_", kind.as_str()),
        _ => fallback_prefix.to_owned(),
    };
    shortuuid(&prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a 12-char id that is not 12 hex chars, or a constant or counter behind `shortuuid`.
    #[test]
    fn a_shortuuid_is_twelve_random_lowercase_hex_characters_after_the_prefix() {
        let id = shortuuid("stt_request_");
        let suffix = id.strip_prefix("stt_request_").expect("prefix is kept verbatim");
        assert_eq!(suffix.len(), 12);
        assert!(
            suffix.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "not lowercase hex: {suffix}"
        );
        assert_ne!(shortuuid(""), shortuuid(""));
    }

    // Catches: R2b on `if session_id`.
    #[test]
    fn an_empty_session_id_uses_the_fallback_prefix() {
        let id = create_inference_request_id(Some(""), RequestKind::Tts, "tts_");
        assert!(id.starts_with("tts_"), "{id}");
        let id = create_inference_request_id(None, RequestKind::Tts, "tts_");
        assert!(id.starts_with("tts_"), "{id}");
    }

    // Catches: dropping the request kind or the separator from the correlation prefix.
    #[test]
    fn a_session_scoped_request_id_is_session_kind_random() {
        let id = create_inference_request_id(Some("inference_abc123"), RequestKind::Eot, "unused_");
        assert!(id.starts_with("inference_abc123_eot_"), "{id}");
        assert_eq!(id.len(), "inference_abc123_eot_".len() + 12);
    }
}
