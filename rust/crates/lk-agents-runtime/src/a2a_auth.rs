//! Credential cryptography at the runtime boundary, using the LiveKit token SDK. The A2A
//! core receives verified typed facts and makes the permission and carrier-selection decisions.

use std::time::Duration;

use livekit_token::{AccessToken, AgentEndpointGrants, AgentThreadGrants, TokenVerifier};
use lk_agents_a2a::auth::{allows, has_subject, selected_token};
use lk_agents_abi::a2a::auth::{
    AuthError, AuthGrant, AuthOperation, AuthPermissions, AuthPrincipal,
    Credentials as CredentialsApi, PrincipalKind,
};

pub struct Credentials {
    api_key: String,
    api_secret: String,
}

fn unauthorized(error: impl std::fmt::Display) -> AuthError {
    AuthError::Unauthorized(format!("invalid token: {error}"))
}

fn grant_from_wire(grant: &AgentThreadGrants) -> AuthGrant {
    AuthGrant {
        all: grant.all,
        scope: grant.scope.clone(),
        thread_ids: grant.thread_ids.clone(),
        list: grant.list,
        create: grant.create,
        write: grant.write,
        read: grant.read,
        delete: grant.delete,
    }
}

fn grant_to_wire(grant: AuthGrant) -> AgentThreadGrants {
    AgentThreadGrants {
        all: grant.all,
        scope: grant.scope,
        thread_ids: grant.thread_ids,
        list: grant.list,
        create: grant.create,
        write: grant.write,
        read: grant.read,
        delete: grant.delete,
    }
}

impl Credentials {
    fn signer(&self, identity: &str, ttl_seconds: u64) -> Result<AccessToken, AuthError> {
        if self.api_key.is_empty() || self.api_secret.is_empty() || ttl_seconds == 0 {
            return Err(AuthError::Unauthorized(
                "credentials and positive token TTL are required".into(),
            ));
        }
        Ok(AccessToken::with_api_key(&self.api_key, &self.api_secret)
            .with_identity(identity)
            .with_ttl(Duration::from_secs(ttl_seconds)))
    }
}

impl CredentialsApi for Credentials {
    fn new(api_key: String, api_secret: String) -> Self {
        Self { api_key, api_secret }
    }

    fn verify(
        &self,
        token: String,
        agent_name: String,
        deployment: String,
    ) -> Result<AuthPrincipal, AuthError> {
        if self.api_key.is_empty() || self.api_secret.is_empty() {
            return Err(AuthError::Unauthorized(
                "the worker holds no verification credentials".into(),
            ));
        }
        let claims = TokenVerifier::with_api_key(&self.api_key, &self.api_secret)
            .verify(&token)
            .map_err(unauthorized)?;
        let grant = grant_from_wire(&claims.agent_thread);
        let endpoint_call = if agent_name.is_empty() {
            claims.agent_endpoint.call
        } else {
            claims.agent_endpoint.allows(&agent_name, &deployment)
        };
        let grant = if has_subject(&grant) {
            Some(grant)
        } else if grant != AuthGrant::default() {
            return Err(AuthError::Unauthorized(
                "an agentThread grant needs all, scope or threadIds".into(),
            ));
        } else {
            None
        };
        if grant.is_none() && !endpoint_call {
            return Err(AuthError::Forbidden(
                "the token grants neither endpoint invocation nor thread access".into(),
            ));
        }
        let identity = claims.sub.clone();
        let kind = if grant.is_some() { PrincipalKind::Grant } else { PrincipalKind::Project };
        let serde_json::Value::Object(claims) =
            serde_json::to_value(claims).map_err(unauthorized)?
        else {
            unreachable!("LiveKit claims serialize as an object")
        };
        Ok(AuthPrincipal { kind, grant, identity, claims, endpoint_call })
    }

    fn principal(
        &self,
        authorization: Option<String>,
        thread_grant: Option<String>,
        node_authorized: bool,
        agent_name: String,
        deployment: String,
    ) -> Result<AuthPrincipal, AuthError> {
        if let Some(token) = selected_token(authorization.as_deref(), thread_grant.as_deref())? {
            return self.verify(token.to_owned(), agent_name, deployment);
        }
        if node_authorized {
            return Ok(AuthPrincipal {
                kind: PrincipalKind::Project,
                grant: None,
                identity: String::new(),
                claims: Default::default(),
                endpoint_call: true,
            });
        }
        Err(AuthError::Unauthorized("a LiveKit bearer token is required".into()))
    }

    fn project_token(&self, identity: String, ttl_seconds: u64) -> Result<String, AuthError> {
        self.signer(&identity, ttl_seconds)?
            .with_agent_endpoint_grants(AgentEndpointGrants { call: true, ..Default::default() })
            .to_jwt()
            .map_err(unauthorized)
    }

    fn thread_grant_token(
        &self,
        grant: AuthGrant,
        identity: String,
        ttl_seconds: u64,
    ) -> Result<String, AuthError> {
        self.signer(&identity, ttl_seconds)?
            .with_agent_thread_grants(grant_to_wire(grant))
            .to_jwt()
            .map_err(unauthorized)
    }
}

impl AuthPermissions for Credentials {
    fn _core_auth_grant_allows(
        grant: AuthGrant,
        operation: AuthOperation,
        thread_id: String,
        scope: Option<String>,
    ) -> bool {
        allows(&grant, operation, &thread_id, scope.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_and_thread_permissions_stay_independent() {
        let credentials = Credentials::new("key".into(), "secret".into());
        let project = credentials.project_token("backend".into(), 3600).unwrap();
        let project = credentials.verify(project, "agent".into(), "prod".into()).unwrap();
        assert!(project.endpoint_call);
        assert!(project.grant.is_none());
        let token = credentials
            .thread_grant_token(
                AuthGrant { scope: "user:42".into(), write: true, ..Default::default() },
                "user:42".into(),
                3600,
            )
            .unwrap();
        let principal = credentials.verify(token, "agent".into(), "prod".into()).unwrap();
        assert!(!principal.endpoint_call);
        assert_eq!(principal.grant.unwrap().scope, "user:42");
    }

    #[test]
    fn signed_room_tokens_never_become_anonymous_endpoint_callers() {
        let credentials = Credentials::new("key".into(), "secret".into());
        let token = AccessToken::with_api_key("key", "secret").to_jwt().unwrap();
        assert!(matches!(
            credentials.verify(token, "agent".into(), "prod".into()),
            Err(AuthError::Forbidden(_))
        ));
        assert!(matches!(
            credentials.principal(None, None, false, "agent".into(), "prod".into()),
            Err(AuthError::Unauthorized(_))
        ));
    }
}
