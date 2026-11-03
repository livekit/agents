//! Authorization decisions over verified claims. Header selection preserves carrier precedence;
//! token cryptography and wall-clock validation execute in runtime.

use lk_agents_abi::a2a::auth::{AuthError, AuthGrant, AuthOperation, AuthPrincipal};

/// A malformed selected carrier refuses, rather than falling back to another credential.
pub fn selected_token<'a>(
    authorization: Option<&'a str>,
    thread_grant: Option<&'a str>,
) -> Result<Option<&'a str>, AuthError> {
    if let Some(header) = authorization {
        let (scheme, token) = header
            .split_once(' ')
            .ok_or_else(|| AuthError::Unauthorized("invalid Authorization header".into()))?;
        if !scheme.eq_ignore_ascii_case("bearer") || token.trim().is_empty() {
            return Err(AuthError::Unauthorized("invalid Authorization header".into()));
        }
        return Ok(Some(token.trim()));
    }
    match thread_grant {
        Some(token) if token.trim().is_empty() => {
            Err(AuthError::Unauthorized("empty thread grant header".into()))
        }
        token => Ok(token.map(str::trim)),
    }
}

/// The console's caller, which nothing authenticates: every permission on every thread, new
/// threads in the `console` scope.
pub fn unauthenticated() -> AuthPrincipal {
    AuthPrincipal {
        kind: lk_agents_abi::a2a::auth::PrincipalKind::Project,
        grant: Some(AuthGrant {
            all: true,
            scope: "console".into(),
            thread_ids: Vec::new(),
            list: true,
            create: true,
            write: true,
            read: true,
            delete: true,
        }),
        identity: "console".into(),
        claims: Default::default(),
        endpoint_call: true,
    }
}

pub fn has_subject(grant: &AuthGrant) -> bool {
    grant.all || !grant.scope.is_empty() || !grant.thread_ids.is_empty()
}

/// A first message must be able to create its scoped thread and write that message.
pub fn allows_first_message(grant: &AuthGrant) -> bool {
    allows(grant, AuthOperation::Create, "", None) && grant.write
}

/// A principal's `operation` on thread `thread_id` in `scope` (`None`: a thread that does not
/// exist), refused without a thread grant or beyond it.
pub fn authorize(
    principal: &AuthPrincipal,
    operation: AuthOperation,
    thread_id: &str,
    scope: Option<&str>,
) -> Result<(), AuthError> {
    let grant = principal.grant.as_ref().ok_or_else(|| {
        AuthError::Forbidden("endpoint call permission does not grant thread access".into())
    })?;
    if allows(grant, operation, thread_id, scope) {
        Ok(())
    } else {
        Err(AuthError::Forbidden(format!("the thread grant does not allow {operation:?}")))
    }
}

/// The scope a message naming no thread creates its thread in, when the grant allows it.
pub fn first_message_scope(principal: &AuthPrincipal) -> Result<&str, AuthError> {
    principal
        .grant
        .as_ref()
        .filter(|grant| allows_first_message(grant))
        .map(|grant| grant.scope.as_str())
        .ok_or_else(|| {
            AuthError::Forbidden(
                "the first message requires scoped create and write authority".into(),
            )
        })
}

pub fn allows(
    grant: &AuthGrant,
    operation: AuthOperation,
    thread_id: &str,
    scope: Option<&str>,
) -> bool {
    match operation {
        AuthOperation::Create => {
            let scope = scope.unwrap_or(&grant.scope);
            grant.create
                && !grant.scope.is_empty()
                && !scope.is_empty()
                && (grant.all || scope == grant.scope)
        }
        AuthOperation::List => {
            grant.list
                && scope.is_some_and(|scope| {
                    grant.all || (!grant.scope.is_empty() && scope == grant.scope)
                })
        }
        AuthOperation::Read | AuthOperation::Write | AuthOperation::Delete => {
            let flag = match operation {
                AuthOperation::Read => grant.read,
                AuthOperation::Write => grant.write,
                AuthOperation::Delete => grant.delete,
                _ => unreachable!(),
            };
            flag && !thread_id.is_empty()
                && (grant.all
                    || grant.thread_ids.iter().any(|id| id == thread_id)
                    || (!grant.scope.is_empty() && scope == Some(grant.scope.as_str())))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorization_header_never_falls_back_to_the_grant_header() {
        assert!(selected_token(Some("Basic denied"), Some("valid")).is_err());
        assert!(selected_token(Some("Bearer "), Some("valid")).is_err());
        assert_eq!(selected_token(Some("Bearer chosen"), Some("other")).unwrap(), Some("chosen"));
        assert_eq!(selected_token(None, Some("valid")).unwrap(), Some("valid"));
        assert_eq!(selected_token(None, None).unwrap(), None);
    }

    #[test]
    fn scope_authorizes_existing_threads_and_create_requires_a_scope_and_each_verb() {
        let mut grant =
            AuthGrant { scope: "user:42".into(), create: true, write: true, ..Default::default() };
        assert!(allows(&grant, AuthOperation::Create, "", None));
        assert!(allows(&grant, AuthOperation::Write, "AT_1", Some("user:42")));
        assert!(!allows(&grant, AuthOperation::Write, "AT_1", Some("user:7")));
        assert!(!allows(&grant, AuthOperation::Read, "AT_1", Some("user:42")));
        grant.all = true;
        grant.scope.clear();
        assert!(!allows(&grant, AuthOperation::Create, "", Some("user:42")));
        grant.write = false;
        assert!(!allows(&grant, AuthOperation::Write, "AT_1", Some("user:42")));
    }
}
