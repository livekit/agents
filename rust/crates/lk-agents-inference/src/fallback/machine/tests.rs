use super::*;

fn at(seconds: u64) -> Ts {
    Ts::from_nanos(seconds * 1_000_000_000)
}
fn fail(attempt: u64) -> ChainEvent {
    ChainEvent::Failed {
        attempt,
        error: ApiError::new(format!("failure {attempt}")).into(),
        produced_output: false,
    }
}
fn request(deadline: Option<Ts>) -> ChainRequest {
    let policy = ChainPolicy {
        max_retry_per_backend: 1,
        retry_interval: Some(0.0),
        ..ChainPolicy::stt_defaults()
    };
    ChainRequest::new(
        Modality::Stt,
        vec![(0, ResourceId(10), "primary".into()), (1, ResourceId(20), "backup".into())],
        policy,
        deadline,
        &FallbackPolicy::default(),
    )
}

#[test]
fn fallback_chain_vectors_execute_allowances_and_retain_every_failure() {
    let mut request = request(None);
    for (attempt, index) in [(1, 0), (2, 0), (3, 1), (4, 1)] {
        let now = Ts::from_nanos((attempt - 1) * 100_000_000);
        assert_eq!(
            request.handle(ChainEvent::Wake, now),
            vec![ChainEffect::SttStreamOpen {
                attempt,
                index,
                stt: ResourceId(if index == 0 { 10 } else { 20 })
            }]
        );
        let effects = request.handle(fail(attempt), now);
        if attempt == 4 {
            let ChainEffect::Finished(Err(error)) = &effects[0] else { panic!("not failed") };
            for i in 1..=4 {
                assert!(error.message().contains(&format!("failure {i}")));
            }
        }
    }
}

#[test]
fn fallback_chain_vectors_never_launch_after_cancel_or_total_expiry() {
    for event in [ChainEvent::Cancel, ChainEvent::Wake] {
        let mut request = request(Some(at(2)));
        request.handle(ChainEvent::Wake, at(0));
        let effects = request.handle(event, at(2));
        assert!(matches!(effects[0], ChainEffect::Cancel { attempt: 1 }));
        assert!(matches!(effects[1], ChainEffect::Finished(Err(_))));
        assert!(request.handle(ChainEvent::Wake, at(3)).is_empty());
    }
}

#[test]
fn fallback_single_health_authority_isolates_instances_and_rejects_stale_probes() {
    let mut health = FallbackPolicy::default();
    let shared = ResourceId(10);
    let other = ResourceId(20);
    health.observe(shared, false, None);
    let probe = health.probe_generation(shared);
    health.observe(shared, false, None);
    assert!(!health.recovered(shared, probe));
    assert!(!health.is_available(shared));
    assert!(health.is_available(other));
    assert!(health.recovered(shared, health.probe_generation(shared)));
    assert!(health.is_available(shared));
}

#[test]
fn fallback_retries_share_the_request_attempt_backoff() {
    let mut request = request(None);
    request.handle(ChainEvent::Wake, at(0));
    assert_eq!(
        request.handle(fail(1), at(0)),
        vec![ChainEffect::WakeAt(Ts::from_nanos(100_000_000))]
    );
    assert!(request.handle(ChainEvent::Wake, Ts::from_nanos(99_000_000)).is_empty());
    assert!(matches!(
        request.handle(ChainEvent::Wake, Ts::from_nanos(100_000_000))[0],
        ChainEffect::SttStreamOpen { index: 0, .. }
    ));
}

#[test]
fn a_host_error_outside_the_api_taxonomy_does_not_retry_the_same_backend() {
    let mut request = request(None);
    request.handle(ChainEvent::Wake, at(0));
    let effects = request.handle(
        ChainEvent::Failed {
            attempt: 1,
            error: ProviderError::Host {
                error: lk_agents_abi::errors::HostError::new("application failure"),
                api_kind: None,
                retryable: true,
                status_code: Some(503),
            },
            produced_output: false,
        },
        at(0),
    );
    assert_eq!(effects, vec![ChainEffect::WakeAt(at(0))]);
    assert!(matches!(
        request.handle(ChainEvent::Wake, at(0))[0],
        ChainEffect::SttStreamOpen { index: 1, .. }
    ));
}

#[test]
fn cooldown_and_probe_recovery_share_one_rule_and_follow_explicit_priority() {
    let primary = ResourceId(900);
    let backup = ResourceId(3);
    let priority = [primary, backup, primary];
    let mut policy = FallbackPolicy::default();
    assert_eq!(policy.candidates(&priority, false), vec![0, 1]);
    policy.observe(primary, false, Some(at(10)));
    let stale_probe = policy.probe_generation(primary);
    policy.observe(primary, false, Some(at(20)));
    assert!(policy.refresh(&priority, at(9)).is_empty());
    assert_eq!(policy.candidates(&priority, false), vec![1]);
    assert_eq!(
        policy.refresh(&priority, at(10)),
        vec![primary],
        "repeated failure does not extend cooldown"
    );
    assert_eq!(
        policy.candidates(&priority, false),
        vec![0, 1],
        "priority is not resource-id order"
    );
    assert!(!policy.recovered(primary, stale_probe), "cooldown recovery invalidates old probes");
    policy.observe(primary, false, None);
    assert!(
        policy.refresh(&priority, at(100)).is_empty(),
        "a backend without cooldown needs a successful probe"
    );
    assert!(policy.recovered(primary, policy.probe_generation(primary)));
    assert_eq!(policy.candidates(&priority, false), vec![0, 1]);
}

#[test]
fn sticky_priority_keeps_the_selected_backend_after_duplicate_entries_are_removed() {
    let primary = ResourceId(900);
    let backup = ResourceId(3);
    let mut policy = FallbackPolicy::default();
    policy.select(backup, 1);
    assert_eq!(policy.candidates(&[primary, primary, backup], true), vec![2, 0]);
}
