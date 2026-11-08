//! Authentication records and the generated credential bridge. Verification and signing execute
//! in runtime; thread and endpoint permission decisions belong to the A2A core.

use serde_json::{Map, Value};

use crate::storage::ThreadInfo;

/// A grant's authoring surface; the native core owns every permission decision.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(
    module = "a2a_auth",
    name = "AgentThreadGrant",
    path = "livekit.agents.a2a",
    dataclass,
    frozen,
    kw_only,
    slots,
    post_init = "auth_grant_post_init",
    emit,
    extract
))]
#[python(
    method(
        name = "can_create",
        args = "scope: str | None = None",
        ret = "bool",
        core = "_core_auth_grant_can_create"
    ),
    method(
        name = "can_list",
        args = "scope: str",
        ret = "bool",
        core = "_core_auth_grant_can_list"
    ),
    method(
        name = "can_read",
        args = "thread: .storage_class.ThreadInfo",
        ret = "bool",
        core = "_core_auth_grant_can_read"
    ),
    method(
        name = "can_write",
        args = "thread: .storage_class.ThreadInfo",
        ret = "bool",
        core = "_core_auth_grant_can_write"
    ),
    method(
        name = "can_delete",
        args = "thread: .storage_class.ThreadInfo",
        ret = "bool",
        core = "_core_auth_grant_can_delete"
    )
)]
pub struct AuthGrant {
    #[host(default = false)]
    pub all: bool,
    #[host(default = "")]
    pub scope: String,
    #[host(annotation = "collections.abc.Sequence[str]", default_factory = "tuple")]
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

/// The verified endpoint caller, with endpoint and thread authority kept independent.
#[derive(Debug, Clone)]
#[crate::host(record(
    module = "a2a_auth",
    name = "Principal",
    path = "livekit.agents.a2a",
    emit,
    extract
))]
pub struct AuthPrincipal {
    pub kind: PrincipalKind,
    pub grant: Option<AuthGrant>,
    pub identity: String,
    pub claims: Map<String, Value>,
    /// An endpoint call grant does not itself permit reading or changing a thread.
    pub endpoint_call: bool,
}

#[crate::host(exceptions(
    module = "a2a_errors",
    path = "livekit.agents.a2a",
    classes(
        AuthError(
            base = "Exception",
            doc = "An HTTP authorization refusal, outside JSON-RPC.",
            attrs(status(ty = "int", value = 401)),
            fields(message(ty = "str", positional)),
            base_args(message)
        ),
        Unauthorized(base = "AuthError", attrs(status(ty = "int", value = 401))),
        Forbidden(base = "AuthError", attrs(status(ty = "int", value = 403)))
    )
))]
const _: () = ();

#[derive(Debug, Clone, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum AuthError {
    #[error("{0}")]
    #[host(kind = "livekit.agents.a2a.Unauthorized")]
    Unauthorized(String),
    #[error("{0}")]
    #[host(kind = "livekit.agents.a2a.Forbidden")]
    Forbidden(String),
}

/// The token of an `Authorization: Bearer <token>` header, trimmed; `None` for another scheme or
/// an empty token.
pub fn bearer_token(header: &str) -> Option<&str> {
    let (scheme, token) = header.split_once(' ')?;
    let token = token.trim();
    (scheme.eq_ignore_ascii_case("bearer") && !token.is_empty()).then_some(token)
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

    /// The caller of a request with these headers: its `Authorization` bearer token, else its
    /// thread grant header (names compared case-insensitively); with neither, the project when
    /// the LiveKit node authorized the call.
    #[host(call)]
    fn principal(
        &self,
        headers: std::collections::BTreeMap<String, String>,
        #[host(kw)] agent_name: String,
        #[host(default = "")] deployment: String,
        #[host(default = false)] node_authorized: bool,
    ) -> Result<AuthPrincipal, AuthError>;

    #[host(call)]
    fn project_token(
        &self,
        #[host(kw, default = "worker")] identity: String,
        #[host(default = 3600)] ttl_seconds: u64,
    ) -> Result<String, AuthError>;

    /// A thread grant for a browser, a device or a delegating agent. It also carries
    /// `agentEndpoint.call`, so a LiveKit node admits it to an endpoint, which authorizes the
    /// thread access itself.
    #[host(call)]
    fn thread_grant_token(
        &self,
        grant: AuthGrant,
        #[host(kw, default = "")] identity: String,
        #[host(default = 3600)] ttl_seconds: u64,
    ) -> Result<String, AuthError>;
}

/// `a2a.Credentials`: the worker's project key pair; signing and verification use the LiveKit
/// SDK.
#[crate::host(facade(path = "livekit.agents.a2a", name = "Credentials", native = Credentials))]
pub trait CredentialsFacade {
    #[host(new)]
    fn new(api_key: String, api_secret: String) -> Self;
    fn verify(
        &self,
        token: String,
        #[host(kw, default = "")] agent_name: String,
        #[host(default = "")] deployment: String,
    ) -> Result<AuthPrincipal, AuthError>;
    /// The caller of a request with these headers (`Credentials.principal`).
    #[host(native = principal)]
    fn _principal(
        &self,
        headers: std::collections::BTreeMap<String, String>,
        #[host(kw)] agent_name: String,
        #[host(default = "")] deployment: String,
        #[host(default = false)] node_authorized: bool,
    ) -> Result<AuthPrincipal, AuthError>;
    fn project_token(
        &self,
        #[host(kw, default = "worker")] identity: String,
        #[host(default = 3600)] ttl_seconds: u64,
    ) -> Result<String, AuthError>;
    /// A thread grant for a browser, a device or a delegating agent. It also carries
    /// `agentEndpoint.call`, so a LiveKit node admits it to an endpoint, which authorizes the
    /// thread access itself.
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

    /// `AgentThreadGrant.can_create(scope)`.
    #[host(function, name = "_core_auth_grant_can_create")]
    fn can_create(grant: AuthGrant, #[host(default = None)] scope: Option<String>) -> bool {
        Self::_core_auth_grant_allows(grant, AuthOperation::Create, String::new(), scope)
    }

    /// `AgentThreadGrant.can_list(scope)`.
    #[host(function, name = "_core_auth_grant_can_list")]
    fn can_list(grant: AuthGrant, scope: String) -> bool {
        Self::_core_auth_grant_allows(grant, AuthOperation::List, String::new(), Some(scope))
    }

    /// `AgentThreadGrant.can_read(thread)`.
    #[host(function, name = "_core_auth_grant_can_read")]
    fn can_read(grant: AuthGrant, thread: ThreadInfo) -> bool {
        Self::_core_auth_grant_allows(
            grant,
            AuthOperation::Read,
            thread.thread_id,
            Some(thread.scope),
        )
    }

    /// `AgentThreadGrant.can_write(thread)`.
    #[host(function, name = "_core_auth_grant_can_write")]
    fn can_write(grant: AuthGrant, thread: ThreadInfo) -> bool {
        Self::_core_auth_grant_allows(
            grant,
            AuthOperation::Write,
            thread.thread_id,
            Some(thread.scope),
        )
    }

    /// `AgentThreadGrant.can_delete(thread)`.
    #[host(function, name = "_core_auth_grant_can_delete")]
    fn can_delete(grant: AuthGrant, thread: ThreadInfo) -> bool {
        Self::_core_auth_grant_allows(
            grant,
            AuthOperation::Delete,
            thread.thread_id,
            Some(thread.scope),
        )
    }

    /// The caller of the console's unauthenticated listener: every thread permission, threads
    /// created in the `console` scope.
    #[host(function)]
    fn _core_unauthenticated_principal() -> AuthPrincipal;
}

#[cfg(test)]
mod tests {
    // Catches: one `Authorization` reader accepting what another refuses: a padded token, another
    // scheme, a scheme's case, or an empty token.
    #[test]
    fn a_bearer_token_is_trimmed_and_only_bearer_counts() {
        use super::bearer_token;
        assert_eq!(bearer_token("Bearer abc"), Some("abc"));
        assert_eq!(bearer_token("bearer  abc "), Some("abc"));
        for header in ["Basic abc", "Bearer", "Bearer   ", "abc"] {
            assert_eq!(bearer_token(header), None, "{header}");
        }
    }
}
