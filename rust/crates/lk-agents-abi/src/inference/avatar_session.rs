//! `inference.AvatarSession`'s native half: the gateway avatar, implemented by the runtime.

use crate::errors::{ApiError, ProviderError};
use crate::{ApiConnectOptions, Extra};

#[crate::host(error(kind = "RuntimeError"))]
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
    /// A creation is in flight or a session was adopted: the first is left untouched.
    #[error(
        "AvatarSession.start() may only be called once per instance; create a new AvatarSession to start another avatar"
    )]
    AlreadyStarted,
    #[error("the avatar session was closed before its provider session was created")]
    Closed,
    #[error(transparent)]
    #[host(delegate)]
    Api(ProviderError),
}

impl From<ApiError> for AvatarError {
    fn from(err: ApiError) -> Self {
        AvatarError::Api(ProviderError::Local(err))
    }
}

/// One avatar provisioned through LiveKit Inference: at most one provider session per instance,
/// created under one operation identity, held until `aclose()` terminates it.
#[crate::host(class(name = "InferenceAvatar"))]
#[async_trait::async_trait]
pub trait InferenceAvatar: Send + Sync + Sized + 'static {
    /// `model` is `"provider"` or `"provider/<avatar_id>"`. The identity defaults to
    /// `"<provider>-inference-avatar"`, the name to the identity. Starts nothing.
    #[host(new)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: String,
        #[host(kw, default = None)] avatar_identity: Option<String>,
        #[host(default = None)] avatar_name: Option<String>,
        #[host(default = None)] extra_kwargs: Option<Extra>,
        base_url: String,
        api_key: String,
        api_secret: String,
        conn_options: ApiConnectOptions,
    ) -> Result<Self, AvatarError>;

    #[host(getter)]
    fn provider(&self) -> String;

    #[host(getter)]
    fn avatar_identity(&self) -> String;

    /// The gateway-generated avatar session id, once created.
    #[host(getter)]
    fn session_id(&self) -> Option<String>;

    /// The provider's own session id while the session is held.
    #[host(getter)]
    fn provider_session_id(&self) -> Option<String>;

    /// Creates the provider session for the agent `agent_identity` in `room_name` and returns the
    /// audio sample rate the avatar takes. A second call while one runs, or once a session was
    /// created, is refused. Cancelling the call stops creating; a session created anyway is
    /// terminated. `livekit_url` defaults to `LIVEKIT_URL`; `headers` are `get_inference_headers()`.
    #[host(await)]
    async fn create(
        &self,
        #[host(kw)] room_name: String,
        room_sid: String,
        #[host(default = None)] livekit_url: Option<String>,
        agent_identity: String,
        headers: Vec<(String, String)>,
    ) -> Result<u32, AvatarError>;

    /// Stops creating and terminates every held session so it stops billing now. Returns once
    /// that cleanup settled or its deadline passed (then logged); a failed terminate keeps the
    /// session held, so a second call retries.
    #[host(await)]
    async fn aclose(&self, headers: Vec<(String, String)>);
}
