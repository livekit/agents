//! The process's telemetry HTTP: one client for every OTLP destination and the recording upload,
//! and the Cloud observability endpoint's requests, signed with the job's bearer, their answer
//! read within a bound and the project owner's recording refusal recognized.

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt as _;
use tokio::sync::OnceCell;

use super::CloudEndpoint;

pub(crate) const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// What an answer's body is read up to: enough for a refusal's text or a `google.rpc.Status`.
const RESPONSE_BYTES: usize = 64 * 1024;

/// The 401 that means "the project owner turned recording off", matched case-insensitively.
/// Other 401s must not disable uploads.
const DISABLED_MARKERS: [&str; 2] = ["data recording is disabled", "disabled by owner"];

/// How an endpoint answered a request it received.
#[derive(Debug)]
pub(crate) enum Answer {
    Sent,
    /// The project disabled recording (Cloud only): the job's other exports stop too.
    Disabled,
    /// Another status, with the start of its body.
    Refused {
        status: u16,
        body: Vec<u8>,
    },
}

/// The process's client: no retries or redirects (each caller owns its attempts).
pub(crate) async fn client() -> Result<reqwest::Client, String> {
    static CLIENT: lk_agents_media::pool::PerProcess<Arc<OnceCell<reqwest::Client>>> =
        lk_agents_media::pool::PerProcess::new(Arc::default);
    let cell = CLIENT.get();
    let client = cell
        .get_or_try_init(|| async {
            crate::tls::config().await?;
            crate::tls::http_client_builder()?
                .retry(reqwest::retry::never())
                .connect_timeout(CONNECT_TIMEOUT)
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|error| error.to_string())
        })
        .await?;
    Ok(client.clone())
}

/// Reads `response` into an [`Answer`]; `cloud` recognizes the owner's refusal.
pub(crate) async fn answer(response: reqwest::Response, cloud: bool) -> Answer {
    let status = response.status();
    if status.is_success() {
        return Answer::Sent;
    }
    let mut body = Vec::new();
    let mut chunks = response.bytes_stream();
    while let Some(Ok(chunk)) = chunks.next().await {
        let room = RESPONSE_BYTES - body.len();
        body.extend_from_slice(&chunk[..chunk.len().min(room)]);
        if body.len() == RESPONSE_BYTES {
            break;
        }
    }
    let status = status.as_u16();
    if cloud && is_disabled(status, &body) {
        return Answer::Disabled;
    }
    Answer::Refused { status, body }
}

fn is_disabled(status: u16, body: &[u8]) -> bool {
    status == 401 && {
        let text = String::from_utf8_lossy(body).to_lowercase();
        DISABLED_MARKERS.iter().any(|m| text.contains(m))
    }
}

/// A request that never reached the endpoint.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Unsent {
    /// The client or the bearer could not be made: retrying does not help.
    #[error("{0}")]
    Setup(String),
    #[error("{0}")]
    Transport(#[from] reqwest::Error),
}

impl CloudEndpoint {
    /// Posts to `/observability/{path}` with the job's bearer; `request` adds the body.
    pub(crate) async fn post(
        &self,
        path: &str,
        request: impl FnOnce(reqwest::RequestBuilder) -> reqwest::RequestBuilder,
    ) -> Result<Answer, Unsent> {
        let client = client().await.map_err(Unsent::Setup)?;
        let bearer = crate::token::observability_bearer(&self.api_key, &self.api_secret)
            .map_err(|error| Unsent::Setup(error.to_string()))?;
        let url = format!("{}/observability/{path}", self.base_url.trim_end_matches('/'));
        let response = request(client.post(url).bearer_auth(bearer)).send().await?;
        Ok(answer(response, true).await)
    }
}

#[cfg(test)]
mod tests {
    // Catches: any 401 disabling uploads, or the owner's refusal missed for its casing.
    #[test]
    fn only_the_owners_refusal_disables_uploads() {
        assert!(super::is_disabled(401, b"Project data recording is DISABLED by owner"));
        assert!(!super::is_disabled(401, b"bad token"));
        assert!(!super::is_disabled(403, b"data recording is disabled"));
    }
}
