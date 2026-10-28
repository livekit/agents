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
        &FallbackHealth::default(),
    )
}

#[test]
fn fallback_chain_vectors_execute_allowances_and_retain_every_failure() {
    let mut request = request(None);
    for (attempt, index) in [(1, 0), (2, 0), (3, 1), (4, 1)] {
        assert_eq!(
            request.handle(ChainEvent::Wake, at(0)),
            vec![ChainEffect::SttStreamOpen {
                attempt,
                index,
                stt: ResourceId(if index == 0 { 10 } else { 20 })
            }]
        );
        let effects = request.handle(fail(attempt), at(0));
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
    let mut health = FallbackHealth::default();
    let shared = ResourceId(10);
    let other = ResourceId(20);
    health.observe(shared, false);
    let probe = health.probe_generation(shared);
    health.observe(shared, false);
    assert!(!health.recovered(shared, probe));
    assert!(!health.is_available(shared));
    assert!(health.is_available(other));
    assert!(health.recovered(shared, health.probe_generation(shared)));
    assert!(health.is_available(shared));
}
