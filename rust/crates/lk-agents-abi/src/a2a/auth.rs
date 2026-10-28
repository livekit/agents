//! Authentication records and the generated credential bridge. Verification and signing execute
//! in runtime; thread and endpoint permission decisions belong to the A2A core.

use serde_json::{Map, Value};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(module = "a2a_auth", emit, extract))]
pub struct AuthGrant {
    #[host(default = false)]
    pub all: bool,
    #[host(default = "")]
    pub scope: String,
    pub thread_ids: Vec<String>,
    #[host(default = false)]
    pub list: bool,
    #[host(default = false)]
    pub create: bool,
    #[host(default = false)]
    pub write: bool,
    #[host(default = false)]
    pub read: bool,
    #[host(default = false)]
    pub delete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[crate::host(record(module = "a2a_auth", emit, extract))]
pub enum AuthOperation {
    Create,
    List,
    Read,
    Write,
    Delete,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "a2a_auth", emit, extract))]
pub enum PrincipalKind {
    Project,
    Grant,
}

#[derive(Debug, Clone)]
#[crate::host(record(module = "a2a_auth", emit, extract))]
pub struct AuthPrincipal {
    pub kind: PrincipalKind,
    pub grant: Option<AuthGrant>,
    pub identity: String,
    pub claims: Map<String, Value>,
    /// An endpoint call grant does not itself permit reading or changing a thread.
    pub endpoint_call: bool,
}

#[derive(Debug, Clone, thiserror::Error)]
#[crate::host(error(with = crate::a2a::auth::auth_error_to_py))]
pub enum AuthError {
    #[error("{0}")]
    Unauthorized(String),
    #[error("{0}")]
    Forbidden(String),
}

#[cfg(feature = "python")]
pub fn auth_error_to_py(error: AuthError) -> pyo3::PyErr {
    let name = match error {
        AuthError::Unauthorized(_) => "Unauthorized",
        AuthError::Forbidden(_) => "Forbidden",
    };
    crate::python::host_exception("livekit.agents.a2a.auth", name, error.to_string())
}

#[crate::host(class(name = "_core_Credentials"))]
pub trait Credentials: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(api_key: String, api_secret: String) -> Self;

    #[host(call)]
    fn verify(
        &self,
        token: String,
        #[host(kw, default = "")] agent_name: String,
        #[host(default = "")] deployment: String,
    ) -> Result<AuthPrincipal, AuthError>;

    #[host(call)]
    fn principal(
        &self,
        authorization: Option<String>,
        thread_grant: Option<String>,
        node_authorized: bool,
        agent_name: String,
        deployment: String,
    ) -> Result<AuthPrincipal, AuthError>;

    #[host(call)]
    fn project_token(
        &self,
        #[host(kw, default = "worker")] identity: String,
        #[host(default = 3600)] ttl_seconds: u64,
    ) -> Result<String, AuthError>;

    #[host(call)]
    fn thread_grant_token(
        &self,
        grant: AuthGrant,
        #[host(kw, default = "")] identity: String,
        #[host(default = 3600)] ttl_seconds: u64,
    ) -> Result<String, AuthError>;
}

#[crate::host(functions)]
pub trait AuthPermissions: Send + Sync + Sized + 'static {
    #[host(function)]
    fn _core_auth_grant_allows(
        grant: AuthGrant,
        operation: AuthOperation,
        thread_id: String,
        scope: Option<String>,
    ) -> bool;
}
