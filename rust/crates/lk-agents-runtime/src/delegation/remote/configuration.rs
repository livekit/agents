//! Immutable native resource configuration. Stable IDs never stand for credentials.

use lk_agents_abi::a2a::client::AgentClientError;

use super::Project;

// Intentionally neither Debug nor serializable: a custom bearer or worker secret never
// reaches a snapshot, generated host record, diagnostic or endpoint-only credential registry.
#[derive(PartialEq, Eq)]
pub(super) struct Configuration {
    endpoint: String,
    credential: Credential,
}

#[derive(PartialEq, Eq)]
enum Credential {
    Anonymous,
    Custom(String),
    Project { project: Project, thread: Option<String> },
}

impl Configuration {
    pub(super) fn new(
        endpoint: &str,
        project: &Project,
        custom_token: Option<&str>,
        thread: Option<&str>,
    ) -> Result<Self, AgentClientError> {
        let credential = match custom_token {
            Some(token) => Credential::Custom(token.into()),
            None => match project.serving(endpoint)? {
                Some(mut project) => {
                    project.url = url::Url::parse(&project.url)
                        .map_err(|error| AgentClientError::InvalidInput(error.to_string()))?
                        .origin()
                        .ascii_serialization();
                    Credential::Project { project, thread: thread.map(str::to_owned) }
                }
                None => Credential::Anonymous,
            },
        };
        Ok(Self { endpoint: endpoint.into(), credential })
    }
}

pub(super) fn endpoint(value: &str) -> Result<String, AgentClientError> {
    let url = url::Url::parse(value)
        .map_err(|error| AgentClientError::InvalidInput(error.to_string()))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(AgentClientError::InvalidInput(
            "a remote endpoint must be an HTTP(S) URL without credentials or a fragment".into(),
        ));
    }
    // A trailing slash in a query is data, not a path separator.
    Ok(if url.query().is_some() {
        url.as_str().into()
    } else {
        url.as_str().trim_end_matches('/').into()
    })
}

pub(super) fn remote_id(endpoint: &str, id: Option<String>) -> Result<String, AgentClientError> {
    match id {
        Some(id) if id.trim().is_empty() => {
            Err(AgentClientError::InvalidInput("a remote agent id must be nonempty".into()))
        }
        Some(id) => Ok(id),
        None => Ok(endpoint.into()),
    }
}

#[cfg(test)]
#[path = "configuration_tests.rs"]
mod tests;
