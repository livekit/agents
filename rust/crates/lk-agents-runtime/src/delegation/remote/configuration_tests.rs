use super::*;

fn project() -> Project {
    Project {
        url: "wss://livekit.test/".into(),
        api_key: "key".into(),
        api_secret: "secret".into(),
        endpoints: String::new(),
    }
}

// Catches: a self-hosted server delegating to its own endpoint on its listener without a token
// (401), because only endpoints on LIVEKIT_URL's origin were signed for.
#[test]
fn the_servers_own_endpoints_are_signed_for_wherever_its_listener_is() {
    let project = Project {
        url: "ws://127.0.0.1:7880".into(),
        endpoints: "http://127.0.0.1:8081/agents/desk/_".into(),
        ..project()
    };
    let own = project.serving("http://127.0.0.1:8081/agents/desk/_/backend").unwrap().unwrap();
    assert_eq!(
        url::Url::parse(&own.url).unwrap().origin().ascii_serialization(),
        "http://127.0.0.1:8081"
    );
    let configured =
        Configuration::new("http://127.0.0.1:8081/agents/desk/_/backend", &project, None, None)
            .unwrap();
    assert!(matches!(configured.credential, Credential::Project { .. }));
    for elsewhere in [
        "http://127.0.0.1:8081/agents/desk/_backup/backend",
        "http://127.0.0.1:8081/agents/other/_/backend",
        "http://127.0.0.1:9000/agents/desk/_/backend",
        "https://booking.test/agents/desk/_/backend",
    ] {
        assert!(project.serving(elsewhere).unwrap().is_none(), "{elsewhere} is not the server's");
    }
    let unsigned = Project { api_secret: String::new(), ..project };
    assert!(unsigned.serving("http://127.0.0.1:8081/agents/desk/_/backend").unwrap().is_none());
}

#[test]
fn default_identity_uses_canonical_url_and_explicit_ids_remain_exact() {
    let target = endpoint("HTTPS://EXAMPLE.TEST:443/expert/").unwrap();
    assert_eq!(target, "https://example.test/expert");
    assert_eq!(remote_id(&target, None).unwrap(), target);
    assert_eq!(remote_id(&target, Some("flights-primary/".into())).unwrap(), "flights-primary/");
    assert!(remote_id(&target, Some("  ".into())).is_err());
    assert!(endpoint("https://user:password@example.test/expert").is_err());
}

#[test]
fn native_configuration_equality_keeps_credentials_and_thread_scope_separate_from_names() {
    let first =
        Configuration::new("https://livekit.test/expert", &project(), None, Some("AT_a")).unwrap();
    let renewed =
        Configuration::new("https://livekit.test/expert", &project(), None, Some("AT_a")).unwrap();
    assert!(first == renewed, "default token renewal has the same configured credentials");
    let other_thread =
        Configuration::new("https://livekit.test/expert", &project(), None, Some("AT_b")).unwrap();
    assert!(first != other_thread);
    let custom =
        Configuration::new("https://livekit.test/expert", &project(), Some("custom-a"), None)
            .unwrap();
    assert!(first != custom);
    let other_custom =
        Configuration::new("https://livekit.test/expert", &project(), Some("custom-b"), None)
            .unwrap();
    assert!(custom != other_custom);
    let other_endpoint =
        Configuration::new("https://livekit.test/elsewhere", &project(), Some("custom-a"), None)
            .unwrap();
    assert!(custom != other_endpoint);
}

#[test]
fn canonical_identity_preserves_a_query_value_ending_in_slash() {
    let target = endpoint("HTTPS://EXAMPLE.TEST:443/expert?scope=/").unwrap();
    assert_eq!(target, "https://example.test/expert?scope=/");
    assert_eq!(remote_id(&target, None).unwrap(), target);
    assert_eq!(
        endpoint("https://example.test/expert/?scope=/").unwrap(),
        "https://example.test/expert/?scope=/"
    );
}
