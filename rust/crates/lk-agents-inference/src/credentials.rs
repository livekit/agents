use lk_agents_abi::errors::Invalid;
use lk_agents_abi::errors::{ApiError, ErrorKind};
use lk_agents_abi::inference::GatewayCredentials;

/// `api_key if is_given(api_key) else os.getenv("LIVEKIT_INFERENCE_API_KEY",
/// os.getenv("LIVEKIT_API_KEY", ""))`. A set-but-empty `LIVEKIT_INFERENCE_API_KEY` shadows
/// `LIVEKIT_API_KEY`, and an explicit `Some("")` is taken as given.
pub fn resolve_credential(
    explicit: Option<&str>,
    inference_env: Option<&str>,
    livekit_env: Option<&str>,
) -> String {
    if let Some(value) = explicit {
        return value.to_owned();
    }
    // `.or()`, not `.filter(|v| !v.is_empty()).or()`: presence, not truthiness.
    inference_env.or(livekit_env).unwrap_or("").to_owned()
}

/// `inference` is `LIVEKIT_INFERENCE_API_KEY` and `server` the server's `LIVEKIT_API_KEY`, as the
/// caller read them.
pub fn resolve_api_key(
    explicit: Option<&str>,
    inference: Option<&str>,
    server: Option<&str>,
) -> Result<String, ApiError> {
    let key = resolve_credential(explicit, inference, server);
    if key.is_empty() {
        return Err(ApiError::new(
            ErrorKind::Auth,
            "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable",
        ));
    }
    Ok(key)
}

/// The same ladder for the secret.
pub fn resolve_api_secret(
    explicit: Option<&str>,
    inference: Option<&str>,
    server: Option<&str>,
) -> Result<String, ApiError> {
    let secret = resolve_credential(explicit, inference, server);
    if secret.is_empty() {
        return Err(ApiError::new(
            ErrorKind::Auth,
            "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental variable",
        ));
    }
    Ok(secret)
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn server(name: &str) -> Option<String> {
    lk_agents_net::env::server_setting(name)
}

/// The ladders, the gateway URL and the identification headers as the host calls them.
pub struct Credentials;

impl lk_agents_abi::inference::InferenceCredentials for Credentials {
    fn inference_api_key(explicit: Option<String>) -> String {
        let (inference, server) = (env("LIVEKIT_INFERENCE_API_KEY"), server("LIVEKIT_API_KEY"));
        resolve_credential(explicit.as_deref(), inference.as_deref(), server.as_deref())
    }

    fn inference_api_secret(explicit: Option<String>) -> String {
        let inference = env("LIVEKIT_INFERENCE_API_SECRET");
        let server = server("LIVEKIT_API_SECRET");
        resolve_credential(explicit.as_deref(), inference.as_deref(), server.as_deref())
    }

    fn inference_url() -> String {
        let (inference, server) = (env("LIVEKIT_INFERENCE_URL"), server("LIVEKIT_URL"));
        crate::headers::default_inference_url(inference.as_deref(), server.as_deref())
    }

    fn inference_credentials(
        base_url: Option<String>,
        api_key: Option<String>,
        api_secret: Option<String>,
    ) -> Result<GatewayCredentials, Invalid> {
        let invalid = |error: ApiError| Invalid(error.message);
        let inference = env("LIVEKIT_INFERENCE_API_KEY");
        let server_key = server("LIVEKIT_API_KEY");
        let key = resolve_api_key(api_key.as_deref(), inference.as_deref(), server_key.as_deref())
            .map_err(invalid)?;
        let inference = env("LIVEKIT_INFERENCE_API_SECRET");
        let server_secret = server("LIVEKIT_API_SECRET");
        let secret = resolve_api_secret(
            api_secret.as_deref(),
            inference.as_deref(),
            server_secret.as_deref(),
        )
        .map_err(invalid)?;
        let url = match base_url.filter(|url| !url.is_empty()) {
            Some(url) => url,
            None => Self::inference_url(),
        };
        Ok(GatewayCredentials { base_url: url, api_key: key, api_secret: secret })
    }

    fn identify_host(user_agent: String) {
        lk_agents_net::job_identity::set_user_agent(user_agent);
    }

    fn inference_identity_headers(
        cx: lk_agents_abi::telemetry::Cx,
        inference_class: Option<String>,
    ) -> crate::headers::Headers {
        crate::headers::job_headers(&cx, inference_class.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: `inference.filter(non_empty).or(livekit)` (an empty inference key must shadow), or
    // an explicit empty string treated as "not given".
    #[test]
    fn explicit_then_inference_then_general_and_empty_counts_as_set() {
        for (explicit, inference, general, want) in [
            (None, Some(""), Some("general"), ""),
            (None, None, Some("general"), "general"),
            (None, Some("specific"), Some("general"), "specific"),
            (None, None, None, ""),
            (Some(""), Some("specific"), Some("general"), ""),
            (Some("mine"), Some("specific"), None, "mine"),
        ] {
            assert_eq!(resolve_credential(explicit, inference, general), want);
        }
    }
}
