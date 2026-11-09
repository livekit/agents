pub use lk_agents_net::strings::shortuuid;

/// The prefix of a gateway connection's session id.
pub const SESSION_ID_PREFIX: &str = "inference_";

/// The prefix of an STT stream's request id when the connection has no session id.
pub const STT_REQUEST_ID_PREFIX: &str = "stt_request_";

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

    // Catches: an id outside the protocol's `guid` format, or a constant or counter behind
    // `shortuuid`.
    #[test]
    fn a_shortuuid_is_the_prefix_then_a_random_protocol_id() {
        let id = shortuuid("stt_request_");
        assert!(lk_agents_net::strings::is_valid_id("stt_request_", &id), "not a guid: {id}");
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
