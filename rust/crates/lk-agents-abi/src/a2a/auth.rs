//! Authentication records and the generated credential bridge. Verification and signing execute
//! in runtime; thread and endpoint permission decisions belong to the A2A core.

use serde_json::{Map, Value};

/// A grant's authoring surface; the native core owns every permission decision.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(
    module = "a2a_auth",
    name = "AgentThreadGrant",
    path = "livekit.agents.a2a.auth",
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
        hook = "auth_grant_can_create"
    ),
    method(name = "can_list", args = "scope: str", ret = "bool", hook = "auth_grant_can_list"),
    method(
        name = "can_read",
        args = "thread: .storage_class.ThreadInfo",
        ret = "bool",
        hook = "auth_grant_can_read"
    ),
    method(
        name = "can_write",
        args = "thread: .storage_class.ThreadInfo",
        ret = "bool",
        hook = "auth_grant_can_write"
    ),
    method(
        name = "can_delete",
        args = "thread: .storage_class.ThreadInfo",
        ret = "bool",
        hook = "auth_grant_can_delete"
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
    path = "livekit.agents.a2a.auth",
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

#[derive(Debug, Clone, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum AuthError {
    #[error("{0}")]
    #[host(kind = "livekit.agents.a2a.auth.Unauthorized")]
    Unauthorized(String),
    #[error("{0}")]
    #[host(kind = "livekit.agents.a2a.auth.Forbidden")]
    Forbidden(String),
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

    /// The caller of the console's unauthenticated listener: every thread permission, threads
    /// created in the `console` scope.
    #[host(function)]
    fn _core_unauthenticated_principal() -> AuthPrincipal;
}
