//! `inference/avatar.py`'s gateway client: the model string, the request body, the session the
//! gateway creates (retried under one idempotency key) and its terminate. The host keeps the room
//! and the agent's audio sink.

use std::sync::Mutex;
use std::time::Duration;

use lk_agents_abi::ApiConnectOptions;
use lk_agents_abi::errors::{ApiError, ProviderError};
use serde_json::{Map, Value};

use crate::headers::{HEADER_INFERENCE_PROVIDER, HEADER_SESSION_ID, Headers};

const LOG: &str = "livekit.agents";

/// The rate when the gateway's answer names none (older gateways); the gateway is authoritative.
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;

/// The whole request's bound; the connect is bounded by the connect options' `timeout`. A gateway
/// that accepts the connection and never answers cannot hang `start()` or `aclose()`.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// `"provider"` or `"provider/<id>"`, trimmed; anything else is refused naming the string.
pub fn parse_model(model: &str) -> Result<(String, Option<String>), AvatarError> {
    let (provider, id) = model.split_once('/').unwrap_or((model, ""));
    let provider = provider.trim_matches(lk_agents_abi::tokenize::is_py_space);
    let id = id.trim_matches(lk_agents_abi::tokenize::is_py_space);
    if provider.is_empty() {
        return Err(AvatarError::InvalidModel(lk_agents_abi::evals::transcript::py_repr(model)));
    }
    Ok((provider.to_owned(), (!id.is_empty()).then(|| id.to_owned())))
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

#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, thiserror::Error)]
pub enum AvatarError {
    #[error("invalid avatar model string: {0} (expected 'provider' or 'provider/<id>')")]
    #[host(kind = "ValueError")]
    InvalidModel(String),
    #[error("pass either a catalog id in the model string ('{0}/<avatar_id>') or image_url, not both")]
    #[host(kind = "ValueError")]
    CatalogIdAndImage(String),
    #[error("livekit_url must be set by argument or the LIVEKIT_URL environment variable")]
    #[host(kind = "ValueError")]
    MissingUrl,
    #[error(
        "AvatarSession.start() may only be called once per instance; create a new AvatarSession to start another avatar"
    )]
    AlreadyStarted,
    #[error(transparent)]
    #[host(delegate)]
    Api(ProviderError),
}

impl From<ApiError> for AvatarError {
    fn from(err: ApiError) -> Self {
        AvatarError::Api(ProviderError::Local(err))
    }
}

/// What the gateway created: the ids `aclose()` terminates with.
#[derive(Debug, Default)]
struct Created {
    session_id: Option<String>,
    provider_session_id: Option<String>,
    /// The HMAC issued with `provider_session_id`; terminate proves this project owns the session.
    terminate_token: Option<String>,
}

/// `_core.InferenceAvatar`: one avatar session provisioned through LiveKit Inference.
pub struct InferenceAvatar {
    provider: String,
    avatar_id: Option<String>,
    options: Map<String, Value>,
    identity: String,
    name: String,
    base_url: String,
    api_key: String,
    api_secret: String,
    conn: ApiConnectOptions,
    created: Mutex<Created>,
    http: std::sync::OnceLock<reqwest::Client>,
}

#[lk_agents_abi::host(object(cell = "arc"))]
impl InferenceAvatar {
    /// `extra_kwargs` is copied, so a later change to the caller's dict does not reach the request.
    /// The identity defaults to `"<provider>-inference-avatar"`, the name to the identity.
    #[host(new(kw_only = "avatar_identity"))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &str,
        #[host(required)] avatar_identity: Option<String>,
        #[host(required)] avatar_name: Option<String>,
        extra_kwargs: Map<String, Value>,
        base_url: String,
        api_key: String,
        api_secret: String,
        timeout: f64,
        max_retry: u32,
        retry_interval: f64,
    ) -> Result<Self, AvatarError> {
        let (provider, avatar_id) = parse_model(model)?;
        if avatar_id.is_some() && extra_kwargs.contains_key("image_url") {
            return Err(AvatarError::CatalogIdAndImage(provider));
        }
        let identity = avatar_identity.unwrap_or_else(|| format!("{provider}-inference-avatar"));
        Ok(InferenceAvatar {
            name: avatar_name.unwrap_or_else(|| identity.clone()),
            identity,
            provider,
            avatar_id,
            options: extra_kwargs,
            base_url,
            api_key,
            api_secret,
            conn: ApiConnectOptions { max_retry, retry_interval, timeout, ..lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS },
            created: Mutex::new(Created::default()),
            http: std::sync::OnceLock::new(),
        })
    }

    #[host(getter)]
    pub fn provider(&self) -> String {
        self.provider.clone()
    }

    #[host(getter)]
    pub fn avatar_identity(&self) -> String {
        self.identity.clone()
    }

    /// The gateway-generated avatar session id, once created.
    #[host(getter)]
    pub fn session_id(&self) -> Option<String> {
        self.created.lock().unwrap().session_id.clone()
    }

    /// The provider's own session id, once created.
    #[host(getter)]
    pub fn provider_session_id(&self) -> Option<String> {
        self.created.lock().unwrap().provider_session_id.clone()
    }

    /// Refuse a second `start()`: one instance is one avatar.
    #[host(method)]
    pub fn check_unstarted(&self) -> Result<(), AvatarError> {
        let created = self.created.lock().unwrap();
        if created.session_id.is_some() || created.provider_session_id.is_some() {
            return Err(AvatarError::AlreadyStarted);
        }
        Ok(())
    }

    /// Create the provider session for the agent `agent_identity` in `room_name`; the gateway
    /// mints the avatar worker's token from these fields. Returns the audio sample rate the
    /// avatar takes. `livekit_url` defaults to `LIVEKIT_URL`; `headers` are
    /// `get_inference_headers()`.
    #[host(method(kw_only = "room_name"))]
    pub async fn create(
        &self,
        room_name: String,
        room_sid: String,
        #[host(required)] livekit_url: Option<String>,
        agent_identity: String,
        headers: Vec<(String, String)>,
    ) -> Result<u32, AvatarError> {
        let livekit_url = livekit_url
            .filter(|url| !url.is_empty())
            .or_else(|| std::env::var("LIVEKIT_URL").ok().filter(|url| !url.is_empty()))
            .ok_or(AvatarError::MissingUrl)?;
        let mut fields = Map::new();
        for (key, value) in [
            ("provider", &self.provider),
            ("livekit_url", &livekit_url),
            ("room_name", &room_name),
            ("room_sid", &room_sid),
            ("avatar_identity", &self.identity),
            ("avatar_name", &self.name),
            ("agent_identity", &agent_identity),
        ] {
            fields.insert(key.into(), value.clone().into());
        }
        let body = payload(fields, self.avatar_id.as_deref(), self.options.clone());
        let answer = self.create_retrying(&body, headers.into_iter().collect()).await?;
        let text = |key: &str| answer.get(key).and_then(Value::as_str).map(str::to_owned);
        let created = Created {
            session_id: text("session_id"),
            provider_session_id: text("provider_session_id"),
            terminate_token: text("terminate_token"),
        };
        let session_id = created.session_id.clone().unwrap_or_default();
        match (&created.provider_session_id, &created.terminate_token) {
            (None, _) => log::warn!(target: LOG, provider = self.provider.as_str(), session_id = session_id.as_str();
                "avatar gateway create response had no provider_session_id; this session cannot be explicitly terminated and will bill until its provider idle timeout"),
            (Some(provider_session_id), None) => log::warn!(target: LOG, provider = self.provider.as_str(), session_id = session_id.as_str(), provider_session_id = provider_session_id.as_str();
                "avatar gateway create response had no terminate_token; this session cannot be explicitly terminated and will bill until its provider idle timeout"),
            _ => {}
        }
        log::debug!(target: LOG, provider = self.provider.as_str(), session_id = session_id.as_str(),
            provider_session_id = created.provider_session_id.as_deref().unwrap_or_default();
            "inference avatar session created");
        *self.created.lock().unwrap() = created;
        let rate = answer.get("sample_rate").and_then(Value::as_u64).filter(|rate| *rate > 0);
        Ok(rate.map_or(DEFAULT_SAMPLE_RATE, |rate| rate as u32))
    }

    /// Terminate the provider session so it stops billing now rather than at its idle timeout.
    /// Best effort: a failure is logged and the ids are kept, so a second call retries.
    #[host(method)]
    pub async fn terminate(&self, headers: Vec<(String, String)>) {
        let (provider_session_id, token) = {
            let created = self.created.lock().unwrap();
            (created.provider_session_id.clone(), created.terminate_token.clone())
        };
        let Some(provider_session_id) = provider_session_id else { return };
        let Some(token) = token else {
            log::debug!(target: LOG, provider = self.provider.as_str(), provider_session_id = provider_session_id.as_str();
                "no terminate_token for this avatar session; skipping explicit terminate");
            return;
        };
        let body = serde_json::json!({
            "provider": self.provider,
            "provider_session_id": provider_session_id,
            "terminate_token": token,
        });
        let mut headers: Headers = headers.into_iter().collect();
        match self.post("/avatar/sessions/terminate", &body, &mut headers).await {
            Ok(_) => {
                let mut created = self.created.lock().unwrap();
                created.provider_session_id = None;
                created.terminate_token = None;
            }
            Err(err) => log::warn!(target: LOG, provider = self.provider.as_str(), provider_session_id = provider_session_id.as_str(), error = err.to_string().as_str();
                "failed to terminate inference avatar session; it will keep billing until its provider idle timeout unless aclose() is called again"),
        }
    }
}

impl InferenceAvatar {
    /// The create call under the connect options: a timeout, a connection failure and a
    /// retryable status are retried, one idempotency key for all, so a retry replays the first
    /// result rather than paying for a second provider session.
    async fn create_retrying(&self, body: &Value, mut headers: Headers) -> Result<Value, ApiError> {
        let key: [u8; 16] = rand::random();
        headers.set("Idempotency-Key", key.iter().map(|b| format!("{b:02x}")).collect::<String>());
        let mut last = None;
        for attempt in 0..=self.conn.max_retry {
            let answer = self.post("/avatar/sessions", body, &mut headers).await.and_then(|text| {
                serde_json::from_str(&text).map_err(|err| PostError::Connection(err.to_string()))
            });
            match answer {
                Ok(answer) => return Ok(answer),
                Err(err) => {
                    let err = match err {
                        PostError::Timeout => {
                            log::warn!(target: LOG, provider = self.provider.as_str(), attempt = attempt;
                                "avatar gateway request timed out");
                            ApiError::timeout(Some(format!(
                                "avatar gateway create timed out after attempt {}",
                                attempt + 1
                            )))
                        }
                        PostError::Connection(message) => {
                            log::warn!(target: LOG, provider = self.provider.as_str(), error = message.as_str();
                                "failed to call avatar gateway");
                            ApiError::connection(Some(message))
                        }
                        PostError::Api(err) if !err.retryable => return Err(err),
                        PostError::Api(err) => {
                            log::warn!(target: LOG, provider = self.provider.as_str(), error = err.message.as_str();
                                "avatar gateway returned a retryable error");
                            err
                        }
                    };
                    last = Some(err);
                }
            }
            if attempt < self.conn.max_retry {
                tokio::time::sleep(Duration::from_secs_f64(self.conn.interval_for_retry(attempt)))
                    .await;
            }
        }
        Err(last.unwrap_or_else(|| {
            ApiError::connection(Some("failed to create avatar session after all retries".into()))
        }))
    }

    /// One POST to the gateway, a fresh token and session id per attempt; the answer's body.
    async fn post(&self, path: &str, body: &Value, headers: &mut Headers) -> Result<String, PostError> {
        let token = crate::token::create_access_token(
            &self.api_key,
            &self.api_secret,
            crate::token::DEFAULT_TTL_SECS,
            crate::token::now_unix(),
        )
        .map_err(|err| PostError::Connection(err.to_string()))?;
        if headers.get(HEADER_SESSION_ID).is_some() {
            headers.set(HEADER_SESSION_ID, crate::ids::shortuuid("inference_"));
        }
        headers.set("Authorization", format!("Bearer {token}"));
        headers.set(HEADER_INFERENCE_PROVIDER, self.provider.clone());
        headers.set("Content-Type", "application/json");
        let url = format!("{}{path}", self.base_url.trim_end_matches('/'));
        let mut request = self.client()?.post(url).timeout(REQUEST_TIMEOUT).body(body.to_string());
        for (name, value) in headers.iter() {
            request = request.header(name, value);
        }
        let response = request.send().await.map_err(PostError::from)?;
        let status = response.status();
        let text = response.text().await.map_err(PostError::from)?;
        if !status.is_success() {
            return Err(PostError::Api(ApiError::from_http(
                &format!("avatar gateway returned an error: {text}"),
                i64::from(status.as_u16()),
                None,
            )));
        }
        Ok(text)
    }

    /// Built on first use, its connect bounded by the connect options' `timeout`.
    fn client(&self) -> Result<&reqwest::Client, PostError> {
        if let Some(client) = self.http.get() {
            return Ok(client);
        }
        crate::transport::install_tls_provider();
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs_f64(self.conn.timeout.max(0.0)))
            .build()
            .map_err(PostError::from)?;
        Ok(self.http.get_or_init(|| client))
    }
}

enum PostError {
    Timeout,
    Connection(String),
    Api(ApiError),
}

impl From<reqwest::Error> for PostError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() { PostError::Timeout } else { PostError::Connection(err.to_string()) }
    }
}

impl std::fmt::Display for PostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PostError::Timeout => f.write_str("timed out"),
            PostError::Connection(message) => f.write_str(message),
            PostError::Api(err) => write!(f, "{err}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

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
    }

    fn avatar(model: &str, base_url: String, max_retry: u32, options: Value) -> Result<InferenceAvatar, AvatarError> {
        InferenceAvatar::new(
            model,
            None,
            None,
            options.as_object().cloned().unwrap_or_default(),
            base_url,
            "key".into(),
            "secret".into(),
            5.0,
            max_retry,
            0.0,
        )
    }

    #[test]
    fn identity_defaults_and_a_catalog_id_excludes_an_image() {
        let av = avatar("lemonslice/agent_1", String::new(), 0, json!({})).unwrap();
        assert_eq!(av.avatar_identity(), "lemonslice-inference-avatar");
        assert_eq!(av.name, "lemonslice-inference-avatar");
        let err = avatar("lemonslice/a", String::new(), 0, json!({"image_url": "u"})).err().unwrap();
        assert!(matches!(err, AvatarError::CatalogIdAndImage(p) if p == "lemonslice"));
    }

    type Seen = Arc<Mutex<Vec<(String, String)>>>;

    /// A gateway answering each request with the next scripted `(status, body)`, recording each
    /// request's head and body.
    async fn gateway(script: Vec<(u16, &'static str)>) -> (String, Seen) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1/", listener.local_addr().unwrap());
        let seen = Seen::default();
        let log = seen.clone();
        tokio::spawn(async move {
            for (status, body) in script {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let n = socket.read(&mut buf).await.unwrap();
                    request.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&request).to_string();
                    let Some((head, rest)) = text.split_once("\r\n\r\n") else { continue };
                    let length = header(head, "content-length").map_or(0, |v| v.parse().unwrap());
                    if rest.len() >= length {
                        log.lock().unwrap().push((head.to_owned(), rest.to_owned()));
                        break;
                    }
                }
                let answer = format!(
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(answer.as_bytes()).await.unwrap();
            }
        });
        (url, seen)
    }

    fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
        head.lines().find_map(|l| {
            let (k, v) = l.split_once(": ")?;
            k.eq_ignore_ascii_case(name).then_some(v)
        })
    }

    #[tokio::test]
    async fn a_retried_create_replays_one_idempotency_key_and_terminate_clears_the_ids() {
        let (url, seen) = gateway(vec![
            (503, "busy"),
            (200, r#"{"session_id":"s","provider_session_id":"p","terminate_token":"t","sample_rate":24000}"#),
            (200, ""),
        ])
        .await;
        let av = avatar("lemonslice/agent_1", url, 3, json!({"prompt": "hi", "custom": 1})).unwrap();
        let identification = vec![("X-LiveKit-Session-ID".to_owned(), "inference_0".to_owned())];
        let rate = av
            .create("room".into(), "RM_1".into(), Some("wss://lk".into()), "agent".into(), identification)
            .await
            .unwrap();
        assert_eq!(rate, 24_000);
        assert_eq!((av.session_id(), av.provider_session_id()), (Some("s".into()), Some("p".into())));
        assert!(matches!(av.check_unstarted(), Err(AvatarError::AlreadyStarted)));
        {
            let seen = seen.lock().unwrap();
            assert_eq!(seen.len(), 2);
            let (first, second) = (&seen[0].0, &seen[1].0);
            assert!(first.starts_with("POST /v1/avatar/sessions "), "{first}");
            assert_eq!(header(first, "idempotency-key"), header(second, "idempotency-key"));
            assert_eq!(header(first, "idempotency-key").unwrap().len(), 32);
            assert_ne!(header(first, "x-livekit-session-id"), header(second, "x-livekit-session-id"));
            assert_eq!(header(first, "x-livekit-inference-provider"), Some("lemonslice"));
            assert!(header(first, "authorization").unwrap().starts_with("Bearer "));
            assert_eq!(
                serde_json::from_str::<Value>(&seen[1].1).unwrap(),
                json!({"provider":"lemonslice","livekit_url":"wss://lk","room_name":"room","room_sid":"RM_1",
                    "avatar_identity":"lemonslice-inference-avatar","avatar_name":"lemonslice-inference-avatar",
                    "agent_identity":"agent","avatar_id":"agent_1","prompt":"hi","extra_kwargs":{"custom":1}})
            );
        }
        av.terminate(Vec::new()).await;
        let seen = seen.lock().unwrap();
        assert!(seen[2].0.starts_with("POST /v1/avatar/sessions/terminate "));
        assert_eq!(header(&seen[2].0, "idempotency-key"), None);
        assert_eq!(
            serde_json::from_str::<Value>(&seen[2].1).unwrap(),
            json!({"provider":"lemonslice","provider_session_id":"p","terminate_token":"t"})
        );
        assert_eq!(av.provider_session_id(), None);
    }

    #[tokio::test]
    async fn a_client_error_is_raised_unretried_and_the_session_can_start_again() {
        let (url, seen) = gateway(vec![(400, "bad image")]).await;
        let av = avatar("lemonslice", url, 3, json!({})).unwrap();
        let err = av
            .create("r".into(), "s".into(), Some("wss://lk".into()), "a".into(), Vec::new())
            .await
            .unwrap_err();
        let AvatarError::Api(ProviderError::Local(err)) = err else { panic!("{err:?}") };
        assert_eq!(err.status_code, Some(400));
        assert!(err.message.starts_with("avatar gateway returned an error: bad image"), "{}", err.message);
        assert_eq!(seen.lock().unwrap().len(), 1);
        assert!(av.check_unstarted().is_ok());
    }
}
