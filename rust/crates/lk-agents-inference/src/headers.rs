pub use lk_agents_net::job_identity::JobIdentity;
pub use lk_agents_net::transport::Headers;

pub const DEFAULT_INFERENCE_URL: &str = "https://agent-gateway.livekit.cloud/v1";
pub const STAGING_INFERENCE_URL: &str = "https://agent-gateway.staging.livekit.cloud/v1";

pub const HEADER_USER_AGENT: &str = "User-Agent";
pub use lk_agents_abi::inference::{
    HEADER_INFERENCE_PRIORITY, HEADER_INFERENCE_PROVIDER, HEADER_SESSION_ID,
};

pub use lk_agents_net::http::{QUOTA_HEADER_FIELDS, extract_quota_usage};

/// [`extract_quota_usage`] over a response's header pairs, names matched case-insensitively.
pub fn quota_usage(headers: Headers) -> Vec<(&'static str, String)> {
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

    if livekit_url.is_some_and(lk_agents_net::strings::is_staging_cloud_url) {
        return STAGING_INFERENCE_URL.to_owned();
    }

    DEFAULT_INFERENCE_URL.to_owned()
}

/// The native identification: `User-Agent` and, for a nonempty class, the priority.
fn native_headers(user_agent: &str, inference_class: Option<&str>) -> Headers {
    let mut headers = Headers::new();
    headers.insert(HEADER_USER_AGENT, user_agent);
    // `if inference_class:` -- an empty class sends no priority header.
    if let Some(class) = inference_class.filter(|class| !class.is_empty()) {
        headers.insert(HEADER_INFERENCE_PRIORITY, class);
    }
    headers
}

/// [`native_headers`] under `user_agent`, then in a job its identity and its own headers.
pub fn identity_headers(
    user_agent: &str,
    inference_class: Option<&str>,
    job: Option<JobIdentity>,
) -> Headers {
    let mut headers = native_headers(user_agent, inference_class);
    let Some(job) = job else {
        return headers;
    };
    let nonempty = |value: Option<String>| value.filter(|value| !value.is_empty());
    for (name, value) in [
        ("X-LiveKit-Room-ID", nonempty(job.room_sid)),
        ("X-LiveKit-Job-ID", nonempty(job.job_id)),
        ("X-LiveKit-Worker-Token", nonempty(job.worker_token)),
        ("X-LiveKit-Agent-ID", nonempty(job.agent_sid)),
    ] {
        if let Some(value) = value {
            headers.insert(name, value);
        }
    }
    for (name, value) in job.headers {
        headers.insert(name, value);
    }
    headers
}

/// The identity of the job `cx` runs for, as a gateway request carries it: [`identity_headers`]
/// under this process's `User-Agent`, outside any job the native headers alone.
pub fn job_headers(cx: &lk_agents_abi::telemetry::Cx, inference_class: Option<&str>) -> Headers {
    use lk_agents_net::job_identity;
    identity_headers(&job_identity::user_agent(), inference_class, job_identity::of(cx.job()))
}

/// What every gateway request carries: the identity of the job `cx` runs for, the pinned
/// provider and the bearer.
pub fn gateway_headers(
    cx: &lk_agents_abi::telemetry::Cx,
    inference_class: Option<&str>,
    provider: Option<&str>,
    bearer: &str,
) -> Headers {
    let mut headers =
        crate::llm::request::request_headers(job_headers(cx, inference_class), provider);
    headers.remove(HEADER_SESSION_ID);
    headers.insert("Authorization", format!("Bearer {bearer}"));
    headers
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

    // Catches: job headers outside a job, an empty id sent, or the job's own headers not winning.
    #[test]
    fn a_job_adds_its_nonempty_identity_and_its_headers() {
        let job = JobIdentity {
            room_sid: Some("RM_1".into()),
            job_id: Some(String::new()),
            agent_sid: Some("PA_1".into()),
            worker_token: None,
            headers: [("X-LiveKit-Room-ID".into(), "RM_2".into()), ("X-Extra".into(), "1".into())]
                .into_iter()
                .collect(),
        };
        let h = identity_headers("ua", Some("low"), Some(job));
        let names: Vec<&str> = h.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            names,
            [
                HEADER_USER_AGENT,
                HEADER_INFERENCE_PRIORITY,
                "X-LiveKit-Room-ID",
                "X-LiveKit-Agent-ID",
                "X-Extra",
            ]
        );
        assert_eq!(h.get("X-LiveKit-Room-ID"), Some("RM_2"));
        assert_eq!(identity_headers("ua", None, None).iter().count(), 1);
    }

    // Catches: R2b on `if inference_class:`.
    #[test]
    fn an_empty_inference_class_sends_no_priority_header() {
        assert_eq!(native_headers("ua", Some("")).get(HEADER_INFERENCE_PRIORITY), None);
        assert_eq!(native_headers("ua", Some("low")).get(HEADER_INFERENCE_PRIORITY), Some("low"));
    }
}
