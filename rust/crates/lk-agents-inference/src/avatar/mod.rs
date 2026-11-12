pub mod gateway;

use lk_agents_abi::inference::avatar_session::AvatarError;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::headers::{Headers, gateway_headers};
use lk_agents_core::request::AttemptFailure;
use lk_agents_net::transport::HttpRequest;

/// A provider session the gateway created.
#[derive(Clone, PartialEq)]
pub struct ProviderSession {
    /// The gateway's id.
    pub session_id: Option<String>,
    /// The provider's own id; terminate names it.
    pub provider_session_id: Option<String>,
    /// The HMAC issued with `provider_session_id`: terminate proves this project owns it.
    pub terminate_token: Option<String>,
    /// The audio rate the avatar takes.
    pub sample_rate: u32,
}

impl ProviderSession {
    /// Without both ids it bills until the provider's idle timeout.
    pub fn terminable(&self) -> bool {
        self.provider_session_id.is_some() && self.terminate_token.is_some()
    }

    /// The gateway replays one operation's answer under its idempotency key.
    fn same_as(&self, other: &ProviderSession) -> bool {
        self.session_id == other.session_id && self.provider_session_id == other.provider_session_id
    }
}

impl std::fmt::Debug for ProviderSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderSession")
            .field("session_id", &self.session_id)
            .field("provider_session_id", &self.provider_session_id)
            .field("terminate_token", &self.terminate_token.as_ref().map(|_| "<redacted>"))
            .field("sample_rate", &self.sample_rate)
            .finish()
    }
}

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
    pub options: GatewayOptions,
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
        let options = GatewayOptions::parse(options)?;
        if avatar_id.is_some() && options.image_url.is_some() {
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

/// The provider options the gateway takes as its own fields; any other is forwarded under
/// `extra_kwargs`, where the gateway's allowlist decides.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GatewayOptions {
    /// The image the avatar is made from; exclusive with a catalog id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idle_prompt: Option<String>,
    /// `idle_timeout`: seconds without input before the provider ends the session.
    #[serde(rename = "idle_timeout_s", skip_serializing_if = "Option::is_none")]
    pub idle_timeout: Option<i64>,
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub extra_kwargs: Map<String, Value>,
}

impl GatewayOptions {
    /// The known options out of `options`, typed; the rest as `extra_kwargs`. A `null` is unset.
    pub fn parse(mut options: Map<String, Value>) -> Result<Self, AvatarError> {
        fn take<T: DeserializeOwned>(
            options: &mut Map<String, Value>,
            key: &str,
        ) -> Result<Option<T>, AvatarError> {
            let Some(value) = options.remove(key).filter(|value| !value.is_null()) else {
                return Ok(None);
            };
            serde_json::from_value(value)
                .map(Some)
                .map_err(|err| AvatarError::InvalidOption(format!("{key:?}: {err}")))
        }
        Ok(GatewayOptions {
            image_url: take(&mut options, "image_url")?,
            prompt: take(&mut options, "prompt")?,
            idle_prompt: take(&mut options, "idle_prompt")?,
            idle_timeout: take(&mut options, "idle_timeout")?,
            extra_kwargs: options,
        })
    }
}

/// `POST /avatar/sessions`.
#[derive(Serialize)]
struct CreateBody<'a> {
    provider: &'a str,
    livekit_url: &'a str,
    room_name: &'a str,
    room_sid: &'a str,
    avatar_identity: &'a str,
    avatar_name: &'a str,
    agent_identity: &'a str,
    /// The catalog avatar; an empty id names none.
    #[serde(skip_serializing_if = "Option::is_none")]
    avatar_id: Option<&'a str>,
    #[serde(flatten)]
    options: &'a GatewayOptions,
}

/// `POST /avatar/sessions/terminate`.
#[derive(Serialize)]
struct TerminateBody<'a> {
    provider: &'a str,
    provider_session_id: &'a str,
    terminate_token: &'a str,
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
    let body = CreateBody {
        provider: &opts.provider,
        livekit_url: &room.livekit_url,
        room_name: &room.room_name,
        room_sid: &room.room_sid,
        avatar_identity: &opts.identity,
        avatar_name: &opts.name,
        agent_identity: &room.agent_identity,
        avatar_id: opts.avatar_id.as_deref().filter(|id| !id.is_empty()),
        options: &opts.options,
    };
    let mut headers = headers;
    headers.insert("Idempotency-Key", idempotency_key);
    request(opts, "/avatar/sessions", &body, headers, bearer)
}

/// The terminate exchange of `session`; `None` when it carries no terminable identity.
pub fn terminate_request(
    opts: &AvatarOptions,
    session: &ProviderSession,
    headers: Headers,
    bearer: &str,
) -> Option<HttpRequest> {
    let body = TerminateBody {
        provider: &opts.provider,
        provider_session_id: session.provider_session_id.as_deref()?,
        terminate_token: session.terminate_token.as_deref()?,
    };
    Some(request(opts, "/avatar/sessions/terminate", &body, headers, bearer))
}

fn request(
    opts: &AvatarOptions,
    path: &str,
    body: &impl Serialize,
    mut headers: Headers,
    bearer: &str,
) -> HttpRequest {
    headers = gateway_headers(Some(&headers), None, Some(&opts.provider), bearer);
    headers.insert("Content-Type", "application/json");
    HttpRequest {
        url: format!("{}{path}", opts.base_url.trim_end_matches('/')),
        headers,
        body: serde_json::to_vec(body).expect("an avatar request body is plain JSON"),
    }
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
        let options = GatewayOptions::parse(object(
            json!({"idle_timeout": 0, "prompt": "", "image_url": null, "other": {"x": true}}),
        ))
        .unwrap();
        assert_eq!(
            serde_json::to_value(options).unwrap(),
            json!({"idle_timeout_s":0, "prompt":"", "extra_kwargs":{"other":{"x":true}}})
        );
        let wrong = GatewayOptions::parse(object(json!({"idle_timeout": "soon"})));
        assert!(matches!(wrong, Err(AvatarError::InvalidOption(_))));
        let err = AvatarOptions::new(
            "lemonslice/a",
            None,
            None,
            object(json!({"image_url": "u"})),
            String::new(),
        );
        assert!(matches!(err, Err(AvatarError::CatalogIdAndImage(p)) if p == "lemonslice"));
    }

    // Catches: the create body missing a room field, an empty catalog id sent, or the options
    // nested instead of beside the room's fields.
    #[test]
    fn the_create_body_carries_the_room_and_the_options_beside_it() {
        let room = AvatarRoom {
            livekit_url: "wss://lk".into(),
            room_name: "r".into(),
            room_sid: "RM_1".into(),
            agent_identity: "agent".into(),
        };
        let opts = AvatarOptions::new(
            "lemonslice/",
            None,
            None,
            object(json!({"prompt": "hi"})),
            "u".into(),
        )
        .unwrap();
        let request = create_request(&opts, &room, Headers::new(), "t", "k");
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(
            body,
            json!({
                "provider": "lemonslice", "livekit_url": "wss://lk", "room_name": "r",
                "room_sid": "RM_1", "avatar_identity": "lemonslice-inference-avatar",
                "avatar_name": "lemonslice-inference-avatar", "agent_identity": "agent",
                "prompt": "hi",
            })
        );
    }

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }
}
