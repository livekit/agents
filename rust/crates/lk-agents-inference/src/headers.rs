//! `inference/_utils.py`'s header/URL half. A job-less caller passes `None`, the reference's
//! `except RuntimeError: pass`.

/// How the host builds the identification headers for one connection: a closure, because
/// `get_inference_headers()` (`_utils.py:76`) mints a fresh session id on every call.
pub type HeaderProvider = std::sync::Arc<dyn Fn() -> Vec<(String, String)> + Send + Sync>;

/// An ordered header map with Python `dict` semantics: replacing a key keeps its position.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Headers(Vec<(String, String)>);

impl Headers {
    pub fn new() -> Self {
        Headers(Vec::new())
    }

    /// Replace in place if present (case-sensitive, like a dict).
    pub fn set(&mut self, name: impl Into<String>, value: impl Into<String>) {
        let name = name.into();
        let value = value.into();
        match self.0.iter_mut().find(|(k, _)| *k == name) {
            Some(slot) => slot.1 = value,
            None => self.0.push((name, value)),
        }
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    pub fn into_vec(self) -> Vec<(String, String)> {
        self.0
    }
}

/// A host's `dict.items()`, in order; a repeated name keeps its first position, as `set` does.
impl FromIterator<(String, String)> for Headers {
    fn from_iter<I: IntoIterator<Item = (String, String)>>(pairs: I) -> Self {
        let mut headers = Headers::new();
        for (name, value) in pairs {
            headers.set(name, value);
        }
        headers
    }
}

pub const DEFAULT_INFERENCE_URL: &str = "https://agent-gateway.livekit.cloud/v1";
pub const STAGING_INFERENCE_URL: &str = "https://agent-gateway.staging.livekit.cloud/v1";

pub const HEADER_USER_AGENT: &str = "User-Agent";
pub const HEADER_SESSION_ID: &str = "X-LiveKit-Session-ID";
pub const HEADER_INFERENCE_PROVIDER: &str = "X-LiveKit-Inference-Provider";
pub const HEADER_INFERENCE_PRIORITY: &str = "X-LiveKit-Inference-Priority";

pub use lk_agents_net::http::{QUOTA_HEADER_FIELDS, extract_quota_usage};

/// [`extract_quota_usage`] over a response's header pairs, names matched case-insensitively.
pub fn quota_usage(headers: Vec<(String, String)>) -> Vec<(&'static str, String)> {
    extract_quota_usage(|name| {
        headers
            .iter()
            .find(|(k, v)| k.eq_ignore_ascii_case(name) && !v.is_empty())
            .map(|(_, v)| v.as_str())
    })
}

/// The default inference URL: `LIVEKIT_INFERENCE_URL`, then staging if `LIVEKIT_URL` names the
/// staging cloud, then production.
pub fn default_inference_url(inference_url: Option<&str>, livekit_url: Option<&str>) -> String {
    // `if inference_url:` -- an empty value falls through.
    if let Some(url) = inference_url
        && !url.is_empty()
    {
        return url.to_owned();
    }

    if livekit_url.unwrap_or_default().contains(".staging.livekit.cloud") {
        return STAGING_INFERENCE_URL.to_owned();
    }

    DEFAULT_INFERENCE_URL.to_owned()
}

/// The native identification: `User-Agent` and, for a nonempty class, the priority.
fn native_headers(user_agent: &str, inference_class: Option<&str>) -> Headers {
    let mut headers = Headers::new();
    headers.set(HEADER_USER_AGENT, user_agent);
    // `if inference_class:` -- an empty class sends no priority header.
    if let Some(class) = inference_class.filter(|class| !class.is_empty()) {
        headers.set(HEADER_INFERENCE_PRIORITY, class);
    }
    headers
}

/// Identification (finished by the host from the job context, else [`native_headers`]) and the
/// pinned provider.
pub fn identification_headers(
    identification: Option<&Headers>,
    inference_class: Option<&str>,
    provider: Option<&str>,
) -> Headers {
    let base = match identification {
        Some(headers) => headers.clone(),
        None => native_headers(&user_agent(env!("CARGO_PKG_VERSION")), inference_class),
    };
    crate::llm::request::request_headers(base, provider)
}

/// [`identification_headers`] and the bearer minted from the key pair.
pub fn gateway_headers(
    identification: Option<&Headers>,
    inference_class: Option<&str>,
    provider: Option<&str>,
    api_key: &str,
    api_secret: &str,
) -> Result<Headers, lk_agents_abi::errors::ApiError> {
    let mut headers = identification_headers(identification, inference_class, provider);
    let token = lk_agents_net::token::inference_bearer(api_key, api_secret)?;
    headers.set("Authorization", format!("Bearer {token}"));
    Ok(headers)
}

/// The job's identification headers (`get_inference_headers()`), bound once by the host that
/// hands the client to a session. Each connection carries them with a fresh session id, as the
/// reference mints one per call; unbound, the native headers.
#[derive(Debug, Default)]
pub struct BoundIdentity(parking_lot::Mutex<Option<Headers>>);

impl BoundIdentity {
    pub fn bind(&self, headers: Headers) {
        *self.0.lock() = Some(headers);
    }

    pub fn for_connection(&self) -> Option<Headers> {
        let mut headers = self.0.lock().clone()?;
        headers.set(HEADER_SESSION_ID, crate::ids::shortuuid(crate::ids::SESSION_ID_PREFIX));
        Some(headers)
    }
}

pub fn user_agent(version: &str) -> String {
    format!("LiveKit Agents/{version} (rust)")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<&'static str> + use<> {
        let owned: Vec<(String, &'static str)> = pairs
            .iter()
            .map(|(k, v)| {
                ((*k).to_owned(), Box::leak(v.to_string().into_boxed_str()) as &'static str)
            })
            .collect();
        move |name: &str| owned.iter().find(|(k, _)| k == name).map(|(_, v)| *v)
    }

    // Catches: `LIVEKIT_INFERENCE_URL=""` treated as set (R2b), or staging matched on a prefix or
    // suffix instead of a substring (the leading dot is part of the needle).
    #[test]
    fn the_default_inference_url() {
        for (env, livekit, want) in [
            (Some(""), None, DEFAULT_INFERENCE_URL),
            (Some(""), Some("wss://x.staging.livekit.cloud"), STAGING_INFERENCE_URL),
            (Some("http://localhost:8080"), None, "http://localhost:8080"),
            (None, Some("wss://myproj.staging.livekit.cloud"), STAGING_INFERENCE_URL),
            (None, Some("wss://xstaging.livekit.cloud"), DEFAULT_INFERENCE_URL),
            (None, None, DEFAULT_INFERENCE_URL),
        ] {
            assert_eq!(default_inference_url(env, livekit), want, "{env:?} {livekit:?}");
        }
    }

    // Catches: reporting a quota dimension the gateway did not stamp, or an empty value (R2b).
    #[test]
    fn only_the_quota_dimensions_the_gateway_stamped_are_reported() {
        let got = extract_quota_usage(map(&[
            ("X-LiveKit-Inference-RPM-Limit", "60"),
            ("X-LiveKit-Inference-RPM-Used", "61"),
            ("X-LiveKit-Inference-TPM-Limit", ""),
        ]));
        assert_eq!(got, vec![("rpm_limit", "60".into()), ("rpm_used", "61".into())]);
        assert!(extract_quota_usage(map(&[])).is_empty());
    }

    // Catches: emitting job headers (including the session ID) with no job context.
    #[test]
    fn with_no_job_context_only_the_user_agent_is_sent() {
        let h = native_headers("ua", None);
        assert_eq!(h.iter().count(), 1);
        assert_eq!(h.get(HEADER_USER_AGENT), Some("ua"));
    }

    // Catches: R2b on `if inference_class:`.
    #[test]
    fn an_empty_inference_class_sends_no_priority_header() {
        assert_eq!(native_headers("ua", Some("")).get(HEADER_INFERENCE_PRIORITY), None);
        assert_eq!(native_headers("ua", Some("low")).get(HEADER_INFERENCE_PRIORITY), Some("low"));
    }
}
