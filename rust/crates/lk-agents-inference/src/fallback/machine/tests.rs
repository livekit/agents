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
    health.observe(shared, false);
    let probe = health.probe_generation(shared);
    health.observe(shared, false);
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

#[cfg(test)]
mod realtime_tests {
    use super::*;

    #[test]
    fn zero_cooldown_still_moves_past_the_failed_model() {
        let mut policy = FallbackPolicy::realtime(2, Duration::ZERO);
        assert_eq!(policy.on_child_error(false, false, Ts::ZERO).swap_to, Some(1));
    }

    fn at(secs: u64) -> Ts {
        Ts::from_nanos(secs * 1_000_000_000)
    }

    // Catches: the rewrite missing, or applied when nothing is left to swap to.
    #[test]
    fn an_unrecoverable_error_is_rewritten_only_while_a_fallback_exists() {
        let mut p = FallbackPolicy::realtime(2, Duration::from_secs(60));
        assert_eq!(
            p.on_child_error(true, false, at(0)),
            ErrorPlan { changes: vec![], rewrite_recoverable: false, swap_to: None },
            "a recoverable error belongs to the plugin's own reconnect"
        );

        let plan = p.on_child_error(false, false, at(1));
        assert_eq!(plan.changes, vec![AvailabilityChange { index: 0, available: false }]);
        assert!(plan.rewrite_recoverable);
        assert_eq!(plan.swap_to, Some(1));

        p.set_active_index(1);
        let plan = p.on_child_error(false, true, at(2));
        assert_eq!(plan.changes, vec![AvailabilityChange { index: 1, available: false }]);
        assert!(!plan.rewrite_recoverable, "every model is down: escalate unchanged");
        assert_eq!(plan.swap_to, None);
    }

    // Catches: swapping while a swap is in flight, or a re-enable the sweep did not report.
    #[test]
    fn the_sweep_reports_each_expired_cooldown_and_a_busy_swap_holds() {
        let mut p = FallbackPolicy::realtime(2, Duration::from_secs(10));
        assert!(p.set_available(0, false, at(0)).is_some());
        assert!(p.set_available(0, false, at(5)).is_none(), "the no-op guard: no re-arm");
        let (target, changes) = p.next_available_index(at(9), false);
        assert_eq!((target, changes), (Some(1), vec![]));
        let (target, changes) = p.next_available_index(at(10), false);
        assert_eq!(target, Some(0), "the primary is preferred once it is back");
        assert_eq!(changes, vec![AvailabilityChange { index: 0, available: true }]);

        let plan = p.on_child_error(false, true, at(11));
        assert!(plan.rewrite_recoverable);
        assert_eq!(plan.swap_to, None, "a swap already in flight is not doubled");
        assert_eq!(p.restart_target(false, at(11)), (0, vec![]));
        assert_eq!(p.restart_target(true, at(11)).0, 1);
    }
}
