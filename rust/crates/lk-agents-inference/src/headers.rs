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
pub const HEADER_ROOM_ID: &str = "X-LiveKit-Room-ID";
pub const HEADER_JOB_ID: &str = "X-LiveKit-Job-ID";
pub const HEADER_AGENT_ID: &str = "X-LiveKit-Agent-ID";
pub const HEADER_SESSION_ID: &str = "X-LiveKit-Session-ID";
pub const HEADER_WORKER_TOKEN: &str = "X-LiveKit-Worker-Token";
pub const HEADER_INFERENCE_PROVIDER: &str = "X-LiveKit-Inference-Provider";
pub const HEADER_INFERENCE_PRIORITY: &str = "X-LiveKit-Inference-Priority";

/// Quota telemetry headers the gateway stamps on LLM completions responses (including 429
/// rejections). Absent means "not enforced / unknown", never zero.
pub const QUOTA_HEADER_FIELDS: [(&str, &str); 6] = [
    ("X-LiveKit-Inference-RPM-Limit", "rpm_limit"),
    ("X-LiveKit-Inference-RPM-Used", "rpm_used"),
    ("X-LiveKit-Inference-TPM-Limit", "tpm_limit"),
    ("X-LiveKit-Inference-TPM-Used", "tpm_used"),
    ("X-LiveKit-Inference-Credits-Limit", "credits_limit"),
    ("X-LiveKit-Inference-Credits-Used", "credits_used"),
];

/// `lookup` should be case-insensitive, like the reference's header maps.
pub fn extract_quota_usage<'a>(
    lookup: impl Fn(&str) -> Option<&'a str>,
) -> Vec<(&'static str, String)> {
    QUOTA_HEADER_FIELDS
        .iter()
        .filter_map(|(header, field)| {
            // An empty header value is dropped, not reported as "".
            match lookup(header) {
                Some(v) if !v.is_empty() => Some((*field, v.to_owned())),
                _ => None,
            }
        })
        .collect()
}

/// [`extract_quota_usage`] over a response's header pairs, names matched case-insensitively.
#[lk_agents_abi::host(export(name = "inference_quota_usage", pid = false))]
pub fn quota_usage(headers: Vec<(String, String)>) -> Vec<(&'static str, String)> {
    extract_quota_usage(|name| {
        headers.iter().find(|(k, v)| k.eq_ignore_ascii_case(name) && !v.is_empty()).map(|(_, v)| v.as_str())
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

#[lk_agents_abi::host(export(name = "inference_default_url", pid = false))]
pub fn default_inference_url_from_env() -> String {
    default_inference_url(
        std::env::var("LIVEKIT_INFERENCE_URL").ok().as_deref(),
        std::env::var("LIVEKIT_URL").ok().as_deref(),
    )
}

/// What `get_inference_headers` reads from the `JobContext`, already filtered by its guards.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JobIdentity {
    /// `ctx.job.room.sid`.
    pub room_id: Option<String>,
    /// `ctx.job.id`.
    pub job_id: Option<String>,
    /// `ctx.agent.sid`, only when `ctx.room.isconnected()`.
    pub agent_id: Option<String>,
    /// `LIVEKIT_WORKER_TOKEN`, hosted agents only.
    pub worker_token: Option<String>,
    /// `ctx.inference_headers`, merged last: the job outranks the model's configuration.
    pub extra: Vec<(String, String)>,
}

/// `job: None` sends only `User-Agent` (+ priority). `session_id` is passed in because the caller
/// reuses it as the request-id prefix.
pub fn inference_headers(
    user_agent: &str,
    inference_class: Option<&str>,
    job: Option<&JobIdentity>,
    session_id: &str,
) -> Headers {
    let mut headers = Headers::new();
    headers.set(HEADER_USER_AGENT, user_agent);

    // `if inference_class:` -- an empty class sends no priority header.
    if let Some(class) = inference_class
        && !class.is_empty()
    {
        headers.set(HEADER_INFERENCE_PRIORITY, class);
    }

    let Some(job) = job else {
        return headers;
    };

    if let Some(room_id) = job.room_id.as_deref() {
        headers.set(HEADER_ROOM_ID, room_id);
    }
    if let Some(job_id) = job.job_id.as_deref() {
        headers.set(HEADER_JOB_ID, job_id);
    }
    if let Some(token) = job.worker_token.as_deref() {
        headers.set(HEADER_WORKER_TOKEN, token);
    }
    if let Some(agent_id) = job.agent_id.as_deref() {
        headers.set(HEADER_AGENT_ID, agent_id);
    }
    for (name, value) in &job.extra {
        headers.set(name, value);
    }
    // Set last: the SDK-owned session ID cannot be replaced by a custom header.
    headers.set(HEADER_SESSION_ID, session_id);

    headers
}

/// Identification (finished by the host from the job context, else [`inference_headers`] with no
/// job), the pinned provider, and the bearer minted from the key pair.
pub(crate) fn gateway_headers(
    identification: Option<&Headers>,
    inference_class: Option<&str>,
    provider: Option<&str>,
    api_key: &str,
    api_secret: &str,
) -> Result<Headers, lk_agents_abi::errors::ApiError> {
    use crate::token::{DEFAULT_TTL_SECS, create_access_token, now_unix};
    let base = match identification {
        Some(headers) => headers.clone(),
        None => inference_headers(&user_agent(env!("CARGO_PKG_VERSION")), inference_class, None, ""),
    };
    let mut headers = crate::llm::request::request_headers(base, provider);
    let token = create_access_token(api_key, api_secret, DEFAULT_TTL_SECS, now_unix())
        .map_err(|e| lk_agents_abi::errors::ApiError::new(e.to_string()))?;
    headers.set("Authorization", format!("Bearer {token}"));
    Ok(headers)
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
        let h = inference_headers("ua", None, None, "inference_abc");
        assert_eq!(h.iter().count(), 1);
        assert_eq!(h.get(HEADER_USER_AGENT), Some("ua"));
        assert_eq!(h.get(HEADER_SESSION_ID), None, "the session id is set inside the try block");
    }

    // Catches: R2b on `if inference_class:`.
    #[test]
    fn an_empty_inference_class_sends_no_priority_header() {
        assert_eq!(
            inference_headers("ua", Some(""), None, "s").get(HEADER_INFERENCE_PRIORITY),
            None
        );
        assert_eq!(
            inference_headers("ua", Some("low"), None, "s").get(HEADER_INFERENCE_PRIORITY),
            Some("low")
        );
    }

    // Catches: merging `ctx.inference_headers` before the model's values, or letting it replace
    // the session ID.
    #[test]
    fn the_job_overrides_the_model_but_never_the_session_id() {
        let job = JobIdentity {
            room_id: Some("RM_1".into()),
            extra: vec![
                (HEADER_ROOM_ID.into(), "RM_override".into()),
                (HEADER_SESSION_ID.into(), "hijacked".into()),
                (HEADER_INFERENCE_PRIORITY.into(), "priority".into()),
            ],
            ..Default::default()
        };
        let h = inference_headers("ua", Some("low"), Some(&job), "inference_real");
        assert_eq!(h.get(HEADER_ROOM_ID), Some("RM_override"));
        assert_eq!(h.get(HEADER_INFERENCE_PRIORITY), Some("priority"));
        assert_eq!(h.get(HEADER_SESSION_ID), Some("inference_real"));
        // and the override replaced in place rather than appending a second Room-ID
        assert_eq!(h.iter().filter(|(k, _)| *k == HEADER_ROOM_ID).count(), 1);
    }

    // Catches: dropping a job header, or emitting Agent-ID when the room is not connected.
    #[test]
    fn all_seven_request_side_livekit_headers_are_reachable() {
        let job = JobIdentity {
            room_id: Some("RM".into()),
            job_id: Some("AJ".into()),
            agent_id: Some("PA".into()),
            worker_token: Some("wt".into()),
            extra: vec![(HEADER_INFERENCE_PROVIDER.into(), "deepgram".into())],
        };
        let h = inference_headers("ua", Some("standard"), Some(&job), "sid");
        let mut names: Vec<&str> =
            h.iter().map(|(k, _)| k).filter(|k| k.starts_with("X-LiveKit-")).collect();
        names.sort_unstable();
        let mut expected = vec![
            HEADER_AGENT_ID,
            HEADER_INFERENCE_PRIORITY,
            HEADER_INFERENCE_PROVIDER,
            HEADER_JOB_ID,
            HEADER_ROOM_ID,
            HEADER_SESSION_ID,
            HEADER_WORKER_TOKEN,
        ];
        expected.sort_unstable();
        assert_eq!(names, expected);
    }
}
