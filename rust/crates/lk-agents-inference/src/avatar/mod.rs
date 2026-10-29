//! `inference/avatar.py`'s gateway avatar: the model string, the create and terminate exchanges
//! and the typed answer ([`ProviderSession`]); [`lifecycle`] owns each session from reservation
//! to adoption or termination. The runtime performs the exchanges; the host keeps the room and
//! the agent's audio sink.

pub mod gateway;
pub mod lifecycle;

use lk_agents_abi::inference::avatar_session::AvatarError;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::headers::{HEADER_INFERENCE_PROVIDER, Headers};
pub use lifecycle::ProviderSession;
use lk_agents_net::attempts::AttemptFailure;
use lk_agents_net::transport::HttpRequest;

/// The rate when the gateway's answer names none (older gateways); the gateway is authoritative.
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;

/// `"provider"` or `"provider/<id>"`, trimmed; anything else is refused naming the string.
pub fn parse_model(model: &str) -> Result<(String, Option<String>), AvatarError> {
    let (provider, id) = model.split_once('/').unwrap_or((model, ""));
    let provider =
        provider.trim_matches(|c| lk_agents_abi::utils::strings::PYTHON_WHITESPACE.contains(&c));
    let id = id.trim_matches(|c| lk_agents_abi::utils::strings::PYTHON_WHITESPACE.contains(&c));
    if provider.is_empty() {
        return Err(AvatarError::InvalidModel(lk_agents_core::text::py_repr(model)));
    }
    Ok((provider.to_owned(), (!id.is_empty()).then(|| id.to_owned())))
}

/// One avatar's gateway configuration, fixed at construction.
#[derive(Debug, Clone)]
pub struct AvatarOptions {
    pub provider: String,
    pub avatar_id: Option<String>,
    /// Provider options: the known ones become the gateway's first-class fields.
    pub options: Map<String, Value>,
    pub identity: String,
    pub name: String,
    pub base_url: String,
}

impl AvatarOptions {
    /// `extra_kwargs` may not name an image beside a catalog id. The identity defaults to
    /// `"<provider>-inference-avatar"`, the name to the identity.
    pub fn new(
        model: &str,
        identity: Option<String>,
        name: Option<String>,
        options: Map<String, Value>,
        base_url: String,
    ) -> Result<Self, AvatarError> {
        let (provider, avatar_id) = parse_model(model)?;
        if avatar_id.is_some() && options.contains_key("image_url") {
            return Err(AvatarError::CatalogIdAndImage(provider));
        }
        let identity = identity.unwrap_or_else(|| format!("{provider}-inference-avatar"));
        Ok(AvatarOptions {
            name: name.unwrap_or_else(|| identity.clone()),
            identity,
            provider,
            avatar_id,
            options,
            base_url,
        })
    }
}

/// Where the avatar joins: the gateway mints the avatar worker's token from these fields.
#[derive(Debug, Clone)]
pub struct AvatarRoom {
    pub livekit_url: String,
    pub room_name: String,
    pub room_sid: String,
    pub agent_identity: String,
}

/// The create exchange of one operation: `idempotency_key` names the operation, so every attempt
/// replays its first result rather than paying for a second provider session. `headers` are the
/// job's identification; `bearer` the gateway token the runtime signed.
pub fn create_request(
    opts: &AvatarOptions,
    room: &AvatarRoom,
    headers: Headers,
    bearer: &str,
    idempotency_key: &str,
) -> HttpRequest {
    let mut fields = Map::new();
    for (key, value) in [
        ("provider", &opts.provider),
        ("livekit_url", &room.livekit_url),
        ("room_name", &room.room_name),
        ("room_sid", &room.room_sid),
        ("avatar_identity", &opts.identity),
        ("avatar_name", &opts.name),
        ("agent_identity", &room.agent_identity),
    ] {
        fields.insert(key.into(), value.clone().into());
    }
    let body = payload(fields, opts.avatar_id.as_deref(), opts.options.clone());
    let mut headers = headers;
    headers.set("Idempotency-Key", idempotency_key);
    request(opts, "/avatar/sessions", &body, headers, bearer)
}

/// The terminate exchange of `session`; `None` when it carries no terminable identity.
pub fn terminate_request(
    opts: &AvatarOptions,
    session: &ProviderSession,
    headers: Headers,
    bearer: &str,
) -> Option<HttpRequest> {
    let body = serde_json::json!({
        "provider": opts.provider,
        "provider_session_id": session.provider_session_id.as_ref()?,
        "terminate_token": session.terminate_token.as_ref()?,
    });
    Some(request(opts, "/avatar/sessions/terminate", &body, headers, bearer))
}

fn request(
    opts: &AvatarOptions,
    path: &str,
    body: &Value,
    mut headers: Headers,
    bearer: &str,
) -> HttpRequest {
    headers.set("Authorization", format!("Bearer {bearer}"));
    headers.set(HEADER_INFERENCE_PROVIDER, opts.provider.clone());
    headers.set("Content-Type", "application/json");
    HttpRequest {
        url: format!("{}{path}", opts.base_url.trim_end_matches('/')),
        headers: headers.into_vec(),
        body: body.to_string().into_bytes(),
    }
}

/// The create request: `fields`, the catalog id, and the options -- the known ones as the
/// gateway's first-class fields, any other under `extra_kwargs` (the gateway's allowlist decides).
pub fn payload(
    mut fields: Map<String, Value>,
    avatar_id: Option<&str>,
    options: Map<String, Value>,
) -> Value {
    if let Some(id) = avatar_id.filter(|id| !id.is_empty()) {
        fields.insert("avatar_id".into(), id.into());
    }
    let mut extra = Map::new();
    for (key, value) in options {
        let target = match key.as_str() {
            "image_url" => Some("image_url"),
            "prompt" => Some("prompt"),
            "idle_prompt" => Some("idle_prompt"),
            "idle_timeout" => Some("idle_timeout_s"),
            _ => None,
        };
        if let Some(target) = target {
            fields.insert(target.into(), value);
        } else {
            extra.insert(key, value);
        }
    }
    if !extra.is_empty() {
        fields.insert("extra_kwargs".into(), extra.into());
    }
    fields.into()
}

/// The create answer's known fields; any other is ignored.
#[derive(Deserialize)]
struct CreateAnswer {
    session_id: Option<String>,
    provider_session_id: Option<String>,
    terminate_token: Option<String>,
    sample_rate: Option<u32>,
}

/// The session a create answer describes; a body that is not the declared shape is malformed.
pub fn decode_created(body: &[u8]) -> Result<ProviderSession, AttemptFailure> {
    let answer: CreateAnswer = serde_json::from_slice(body)
        .map_err(|err| AttemptFailure::Malformed(format!("avatar create answer: {err}")))?;
    Ok(ProviderSession {
        session_id: answer.session_id,
        provider_session_id: answer.provider_session_id,
        terminate_token: answer.terminate_token,
        sample_rate: answer.sample_rate.filter(|rate| *rate > 0).unwrap_or(DEFAULT_SAMPLE_RATE),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Catches: a model string split wrong, a whitespace-only provider accepted, or a known
    // option sent under `extra_kwargs` (the gateway would drop it).
    #[test]
    fn model_string_and_request_options() {
        assert_eq!(
            parse_model(" lemonslice / id/x ").unwrap(),
            ("lemonslice".into(), Some("id/x".into()))
        );
        assert_eq!(
            parse_model(" /id").unwrap_err().to_string(),
            "invalid avatar model string: ' /id' (expected 'provider' or 'provider/<id>')"
        );
        assert_eq!(parse_model("lemonslice/").unwrap(), ("lemonslice".into(), None));
        let options = json!({"idle_timeout": 0, "prompt": "", "other": {"x": true}});
        assert_eq!(
            payload(Map::new(), Some("id"), options.as_object().unwrap().clone()),
            json!({"avatar_id":"id", "idle_timeout_s":0, "prompt":"", "extra_kwargs":{"other":{"x":true}}})
        );
        assert_eq!(payload(Map::new(), Some(""), Map::new()), json!({}));
        let err = AvatarOptions::new(
            "lemonslice/a",
            None,
            None,
            json!({"image_url": "u"}).as_object().unwrap().clone(),
            String::new(),
        );
        assert!(matches!(err, Err(AvatarError::CatalogIdAndImage(p)) if p == "lemonslice"));
    }

    // Catches: a session-id typed as anything but a string accepted, or the gateway's rate
    // replaced by the default.
    #[test]
    fn the_create_answer_is_decoded_to_its_typed_fields() {
        let session = decode_created(br#"{"session_id":"s","provider_session_id":"p","terminate_token":"t","sample_rate":24000,"x":1}"#).unwrap();
        assert_eq!(
            (session.session_id.as_deref(), session.sample_rate, session.terminable()),
            (Some("s"), 24_000, true)
        );
        assert_eq!(decode_created(b"{}").unwrap().sample_rate, DEFAULT_SAMPLE_RATE);
        assert!(matches!(
            decode_created(br#"{"session_id":7}"#),
            Err(AttemptFailure::Malformed(_))
        ));
    }
}
