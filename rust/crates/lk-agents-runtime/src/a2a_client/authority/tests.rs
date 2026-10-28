use super::*;
use std::time::Duration;

use livekit_token::{AccessToken, AgentThreadGrants};
use lk_agents_a2a::delegation::{InteractionId, RemoteInteraction};

const ENDPOINT: &str = "https://project.livekit.cloud/agents/expert/production/help";
const KEY: &str = "delegation-test-project";
const SECRET: &str = "delegation-test-project-secret";
const THREAD: &str = "AT_caller";
const CALLER: &str = "frontdesk";

fn grants() -> AgentThreadGrants {
    AgentThreadGrants {
        thread_ids: vec![THREAD.into()],
        read: true,
        write: true,
        ..AgentThreadGrants::default()
    }
}

fn token(
    key: &str,
    secret: &str,
    identity: &str,
    grants: AgentThreadGrants,
    ttl: Duration,
) -> String {
    AccessToken::with_api_key(key, secret)
        .with_identity(identity)
        .with_agent_thread_grants(grants)
        .with_ttl(ttl)
        .to_jwt()
        .unwrap()
}

fn assertion(token: &str) -> AuthorityAssertion<'_> {
    AuthorityAssertion::Grant {
        project_url: "https://project.livekit.cloud",
        api_key: KEY,
        api_secret: SECRET,
        token,
    }
}

#[test]
fn delegation_authority_matrix_shares_only_after_local_grant_verification() {
    let valid = token(KEY, SECRET, CALLER, grants(), Duration::from_secs(3600));
    let proof = verify(ENDPOINT, THREAD, CALLER, assertion(&valid), SystemTime::now()).unwrap();
    let mut interaction = RemoteInteraction::new(InteractionId {
        delegation_id: "delegation-1".into(),
        tool_id: "expert".into(),
        parent_call_id: "call-1".into(),
        endpoint: ENDPOINT.into(),
    })
    .unwrap();
    let outgoing = interaction
        .message(
            "check coverage".into(),
            "m1".into(),
            Some(THREAD),
            CALLER,
            Some(&proof),
            false,
            None,
        )
        .unwrap();
    assert_eq!(outgoing.context_id.as_deref(), Some(THREAD));
    assert_eq!(
        verify(
            "https://foreign.livekit.cloud/agents/expert/production/help",
            THREAD,
            CALLER,
            assertion(&valid),
            SystemTime::now()
        )
        .unwrap_err(),
        AuthorityError::WrongProject,
    );

    let no_read = AgentThreadGrants { read: false, ..grants() };
    let no_write = AgentThreadGrants { write: false, ..grants() };
    let wrong_thread = AgentThreadGrants { thread_ids: vec!["AT_other".into()], ..grants() };
    let scope_only = AgentThreadGrants { thread_ids: vec![], scope: "user:42".into(), ..grants() };
    let ttl = Duration::from_secs(3600);
    let invalid = [
        (token(KEY, "forged-secret", CALLER, grants(), ttl), AuthorityError::InvalidToken),
        (token("foreign-project", SECRET, CALLER, grants(), ttl), AuthorityError::InvalidToken),
        (token(KEY, SECRET, CALLER, no_read, ttl), AuthorityError::ThreadDenied),
        (token(KEY, SECRET, CALLER, no_write, ttl), AuthorityError::ThreadDenied),
        (token(KEY, SECRET, CALLER, wrong_thread, ttl), AuthorityError::ThreadDenied),
        (token(KEY, SECRET, CALLER, scope_only, ttl), AuthorityError::ThreadDenied),
        (token(KEY, SECRET, "another-caller", grants(), ttl), AuthorityError::WrongCaller),
        (token(KEY, SECRET, "", grants(), ttl), AuthorityError::WrongCaller),
        (token(KEY, SECRET, CALLER, grants(), Duration::ZERO), AuthorityError::InactiveToken),
    ];
    for (bearer, expected) in invalid {
        let result = verify(ENDPOINT, THREAD, CALLER, assertion(&bearer), SystemTime::now());
        assert_eq!(result.unwrap_err(), expected);
    }
    assert_eq!(
        verify(ENDPOINT, "", CALLER, assertion(&valid), SystemTime::now()).unwrap_err(),
        AuthorityError::MissingCaller
    );
    assert_eq!(
        verify(ENDPOINT, THREAD, "", assertion(&valid), SystemTime::now()).unwrap_err(),
        AuthorityError::MissingCaller
    );
}

#[test]
fn configured_delegation_trust_covers_exact_canonical_destination_thread_and_identity() {
    let trust = ConfiguredTrust { endpoint: ENDPOINT, thread_id: THREAD, caller_identity: CALLER };
    let assertion = || AuthorityAssertion::ConfiguredTrust(trust);
    let canonical = "https://PROJECT.livekit.cloud:443/agents/expert/production/help/";
    let proof = verify(canonical, THREAD, CALLER, assertion(), SystemTime::now()).unwrap();
    let mut interaction = RemoteInteraction::new(InteractionId {
        delegation_id: "delegation-1".into(),
        tool_id: "expert".into(),
        parent_call_id: "call-1".into(),
        endpoint: canonical.into(),
    })
    .unwrap();
    assert_eq!(
        interaction
            .message("request".into(), "m1".into(), Some(THREAD), CALLER, Some(&proof), true, None,)
            .unwrap()
            .context_id
            .as_deref(),
        Some(THREAD)
    );

    for (destination, thread, caller) in [
        ("https://attacker.invalid/agents/expert/production/help", THREAD, CALLER),
        ("https://project.livekit.cloud/agents/other/production/help", THREAD, CALLER),
        ("http://project.livekit.cloud/agents/expert/production/help", THREAD, CALLER),
        ("https://project.livekit.cloud:8443/agents/expert/production/help", THREAD, CALLER),
        (ENDPOINT, "AT_other", CALLER),
        (ENDPOINT, THREAD, "other-caller"),
    ] {
        assert_eq!(
            verify(destination, thread, caller, assertion(), SystemTime::now()).unwrap_err(),
            AuthorityError::TrustMismatch
        );
    }
    for destination in [
        "https://user:password@project.livekit.cloud/agents/expert/production/help",
        "https://project.livekit.cloud/agents/expert/production/help?other=1",
        "https://project.livekit.cloud/agents/expert/production/help#fragment",
        "file:///agents/expert/production/help",
    ] {
        assert_eq!(
            verify(destination, THREAD, CALLER, assertion(), SystemTime::now()).unwrap_err(),
            AuthorityError::InvalidEndpoint
        );
    }
}
