use super::*;

const NOW: Duration = Duration::ZERO;

fn core(idle: usize, capacity: usize, pending: usize) -> PoolState {
    PoolState::new(PoolLimits {
        idle_processes: idle,
        concurrent_initializations: capacity,
        pending_jobs: pending,
        cleanup_timeout: Duration::from_secs(3),
    })
}

fn admit(core: &mut PoolState, id: u64) -> Vec<PoolEffect> {
    core.handle(PoolInput::Admit { request: ReservationId(id), job_id: format!("job-{id}") }, NOW)
}

#[test]
fn pool_allocation_phase_matrix() {
    let mut pool = core(1, 1, 2);
    assert_eq!(pool.handle(PoolInput::Start, NOW), [PoolEffect::Create(ProcessId(0))]);
    admit(&mut pool, 0);
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Creating);
    assert_eq!(pool.reservation(ReservationId(0)), Some(ReservationState::Waiting));
    pool.handle(PoolInput::Created(ProcessId(0)), NOW);
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Initializing);
    assert_eq!(pool.reservation(ReservationId(0)), Some(ReservationState::Waiting));
    pool.handle(PoolInput::Ready(ProcessId(0)), NOW);
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Reserved);
    assert_eq!(pool.reservation(ReservationId(0)), Some(ReservationState::Reserved(ProcessId(0))));
    admit(&mut pool, 1);
    assert_eq!(pool.reservation(ReservationId(1)), Some(ReservationState::Waiting));
    pool.handle(
        PoolInput::LaunchSucceeded { request: ReservationId(0), process: ProcessId(0) },
        NOW,
    );
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Running);
    pool.handle(PoolInput::Exited(ProcessId(0)), NOW);
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Retiring);
    assert_eq!(pool.reservation(ReservationId(1)), Some(ReservationState::Waiting));
    pool.handle(PoolInput::Cleaned(ProcessId(0)), NOW);
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Cleaned);
    assert_eq!(pool.reservation(ReservationId(1)), Some(ReservationState::Waiting));

    let mut warmed = core(1, 1, 1);
    warmed.handle(PoolInput::Start, NOW);
    warmed.handle(PoolInput::Created(ProcessId(0)), NOW);
    warmed.handle(PoolInput::Ready(ProcessId(0)), NOW);
    assert_eq!(warmed.phase(ProcessId(0)), ProcessPhase::Ready);
    warmed.handle(PoolInput::Exited(ProcessId(0)), NOW);
    admit(&mut warmed, 0);
    assert_eq!(warmed.ready(), 0, "dead warm entries cannot remain eligible");
    assert_eq!(warmed.reservation(ReservationId(0)), Some(ReservationState::Waiting));
}

#[test]
fn pool_pending_admission_is_fifo_bounded_and_closed_by_drain() {
    let mut pool = core(0, 1, 2);
    pool.handle(PoolInput::Start, NOW);
    assert_eq!(admit(&mut pool, 0), [PoolEffect::Create(ProcessId(0))]);
    assert!(admit(&mut pool, 1).is_empty(), "initialization admission precedes tasks");
    assert_eq!(
        admit(&mut pool, 2),
        [PoolEffect::Refused { request: ReservationId(2), reason: PoolAdmissionError::Full }]
    );
    pool.handle(PoolInput::Created(ProcessId(0)), NOW);
    pool.handle(PoolInput::Ready(ProcessId(0)), NOW);
    assert_eq!(pool.reservation(ReservationId(0)), Some(ReservationState::Reserved(ProcessId(0))));
    assert_eq!(pool.reservation(ReservationId(1)), Some(ReservationState::Waiting));
    pool.handle(PoolInput::Drain, NOW);
    assert_eq!(
        admit(&mut pool, 3),
        [PoolEffect::Refused { request: ReservationId(3), reason: PoolAdmissionError::Draining }]
    );
    pool.handle(PoolInput::Close, NOW);
    assert_eq!(
        admit(&mut pool, 4),
        [PoolEffect::Refused { request: ReservationId(4), reason: PoolAdmissionError::Closed }]
    );
}

#[test]
fn pool_reservation_failure_cancel_releases_capacity_and_exhausts_once() {
    let mut pool = core(0, 1, 2);
    pool.handle(PoolInput::Start, NOW);
    admit(&mut pool, 0);
    pool.handle(PoolInput::Created(ProcessId(0)), NOW);
    pool.handle(PoolInput::Ready(ProcessId(0)), NOW);
    pool.handle(PoolInput::LaunchFailed { request: ReservationId(0), process: ProcessId(0) }, NOW);
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Retiring);
    assert_eq!(pool.reservation(ReservationId(0)), Some(ReservationState::Waiting));
    let effects = pool.handle(PoolInput::Cleaned(ProcessId(0)), NOW);
    assert_eq!(effects, [PoolEffect::Create(ProcessId(1))]);
    pool.handle(PoolInput::Cancel(ReservationId(0)), NOW);
    admit(&mut pool, 1);
    pool.handle(PoolInput::Created(ProcessId(1)), NOW);
    pool.handle(PoolInput::Ready(ProcessId(1)), NOW);
    assert_eq!(pool.reservation(ReservationId(1)), Some(ReservationState::Reserved(ProcessId(1))));

    let mut failed = core(0, 1, 2);
    failed.handle(PoolInput::Start, NOW);
    admit(&mut failed, 0);
    for process in 0..MAX_ACQUIRE_ATTEMPTS {
        failed.handle(PoolInput::SpawnFailed(ProcessId(process.into())), NOW);
        failed.handle(PoolInput::Cleaned(ProcessId(process.into())), NOW);
    }
    assert_eq!(
        failed.reservation(ReservationId(0)),
        Some(ReservationState::Failed(PoolFailure::AcquireExhausted {
            attempts: MAX_ACQUIRE_ATTEMPTS
        }))
    );
    assert_eq!(failed.waiting(), 0);
    assert!(failed.handle(PoolInput::Wake, Duration::from_secs(1)).is_empty());
}

#[test]
fn pool_launch_retries_have_their_own_allowance() {
    let mut pool = core(0, 1, 1);
    pool.handle(PoolInput::Start, NOW);
    admit(&mut pool, 0);
    for attempt in 0..MAX_LAUNCH_ATTEMPTS {
        let process = ProcessId(attempt.into());
        pool.handle(PoolInput::Created(process), NOW);
        pool.handle(PoolInput::Ready(process), NOW);
        assert_eq!(pool.reservation(ReservationId(0)), Some(ReservationState::Reserved(process)));
        pool.handle(PoolInput::LaunchFailed { request: ReservationId(0), process }, NOW);
        pool.handle(PoolInput::Cleaned(process), NOW);
    }
    assert_eq!(
        pool.reservation(ReservationId(0)),
        Some(ReservationState::Failed(PoolFailure::LaunchExhausted {
            attempts: MAX_LAUNCH_ATTEMPTS
        }))
    );
}

#[test]
fn releasing_a_failed_request_cannot_erase_a_new_jobs_identity() {
    let mut pool = core(0, 1, 3);
    pool.handle(PoolInput::Start, NOW);
    admit(&mut pool, 0);
    for process in 0..MAX_ACQUIRE_ATTEMPTS {
        pool.handle(PoolInput::SpawnFailed(ProcessId(process.into())), NOW);
        pool.handle(PoolInput::Cleaned(ProcessId(process.into())), NOW);
    }
    let replacement = PoolInput::Admit { request: ReservationId(1), job_id: "job-0".into() };
    pool.handle(replacement, NOW);
    pool.handle(PoolInput::Cancel(ReservationId(0)), NOW);
    assert_eq!(
        pool.handle(PoolInput::Admit { request: ReservationId(2), job_id: "job-0".into() }, NOW),
        [PoolEffect::Refused {
            request: ReservationId(2),
            reason: PoolAdmissionError::DuplicateJob,
        }]
    );
}

#[test]
fn pool_cleanup_deadline_observes_incomplete_ownership_once() {
    let mut pool = core(1, 1, 1);
    pool.handle(PoolInput::Start, NOW);
    pool.handle(PoolInput::Close, NOW);
    assert_eq!(pool.wake_at(), Some(Duration::from_secs(3)));
    assert!(pool.handle(PoolInput::Wake, Duration::from_secs(2)).is_empty());
    assert_eq!(
        pool.handle(PoolInput::Wake, Duration::from_secs(3)),
        [PoolEffect::CleanupIncomplete(PoolCleanupObservation {
            deadline: Duration::from_secs(3),
            processes: vec![ProcessId(0)],
        })]
    );
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Retiring);
    assert!(pool.handle(PoolInput::Close, Duration::from_secs(4)).is_empty());
    assert_eq!(pool.wake_at(), None, "repeated close does not reset the deadline");
    assert_eq!(
        pool.handle(PoolInput::Created(ProcessId(0)), Duration::from_secs(4)),
        [PoolEffect::Close(ProcessId(0))]
    );
    pool.handle(PoolInput::Cleaned(ProcessId(0)), Duration::from_secs(4));
    assert_eq!(
        pool.handle(PoolInput::CleanupComplete, Duration::from_secs(4)),
        [PoolEffect::CloseComplete]
    );
}

#[test]
fn an_older_initialization_failure_cannot_release_a_newer_reservation() {
    let mut pool = core(2, 2, 1);
    pool.handle(PoolInput::Start, NOW);
    admit(&mut pool, 0);
    pool.handle(PoolInput::Created(ProcessId(0)), NOW);
    pool.handle(PoolInput::Created(ProcessId(1)), NOW);
    pool.handle(PoolInput::Ready(ProcessId(1)), NOW);
    assert_eq!(pool.reservation(ReservationId(0)), Some(ReservationState::Reserved(ProcessId(1))));
    pool.handle(PoolInput::SpawnFailed(ProcessId(0)), NOW);
    assert_eq!(pool.reservation(ReservationId(0)), Some(ReservationState::Reserved(ProcessId(1))));
}

#[test]
fn cleanup_due_wake_precedes_same_instant_completion() {
    let mut pool = core(0, 1, 1);
    pool.handle(PoolInput::Start, NOW);
    pool.handle(PoolInput::Close, NOW);
    assert_eq!(
        pool.handle(PoolInput::CleanupComplete, Duration::from_secs(3)),
        [
            PoolEffect::CleanupIncomplete(PoolCleanupObservation {
                deadline: Duration::from_secs(3),
                processes: vec![],
            }),
            PoolEffect::CloseComplete,
        ]
    );
}

#[test]
fn join_failure_retries_once_per_deadline_and_releases_only_on_a_receipt() {
    let mut pool = core(0, 1, 1);
    pool.handle(PoolInput::Start, NOW);
    admit(&mut pool, 0);
    pool.handle(PoolInput::Created(ProcessId(0)), NOW);
    pool.handle(PoolInput::Close, NOW);
    assert_eq!(pool.handle(PoolInput::Closed(ProcessId(0)), NOW), [PoolEffect::Join(ProcessId(0))]);
    assert!(pool.handle(PoolInput::JoinFailed(ProcessId(0)), NOW).is_empty());
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Retiring);
    assert_eq!(pool.wake_at(), Some(REFILL_INTERVAL));
    assert_eq!(pool.handle(PoolInput::Wake, REFILL_INTERVAL), [PoolEffect::Join(ProcessId(0))]);
    assert!(
        pool.handle(PoolInput::Wake, REFILL_INTERVAL * 2).is_empty(),
        "a pending join cannot accumulate duplicate retry tasks"
    );
    assert_eq!(
        pool.handle(
            PoolInput::Joined { process: ProcessId(0), outcome: PoolJoinOutcome::NotLaunched },
            REFILL_INTERVAL * 2
        ),
        [PoolEffect::Released { process: ProcessId(0), outcome: PoolJoinOutcome::NotLaunched }]
    );
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Cleaned);
}

#[test]
fn a_reaping_receipt_does_not_skip_the_owned_close_callback() {
    let mut pool = core(1, 1, 1);
    pool.handle(PoolInput::Start, NOW);
    pool.handle(PoolInput::Created(ProcessId(0)), NOW);
    pool.handle(PoolInput::Ready(ProcessId(0)), NOW);
    assert_eq!(
        pool.handle(
            PoolInput::Joined { process: ProcessId(0), outcome: PoolJoinOutcome::Reaped },
            NOW
        ),
        [PoolEffect::Close(ProcessId(0)), PoolEffect::Create(ProcessId(1))]
    );
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Retiring);
    assert_eq!(
        pool.handle(PoolInput::Closed(ProcessId(0)), NOW),
        [PoolEffect::Released { process: ProcessId(0), outcome: PoolJoinOutcome::Reaped }]
    );
}

#[test]
fn exit_observation_failure_retains_cleanup_and_retries_without_fabricating_a_receipt() {
    let mut pool = core(1, 1, 1);
    pool.handle(PoolInput::Start, NOW);
    pool.handle(PoolInput::Created(ProcessId(0)), NOW);
    pool.handle(PoolInput::Ready(ProcessId(0)), NOW);
    assert_eq!(
        pool.handle(PoolInput::ExitFailed(ProcessId(0)), NOW),
        [PoolEffect::Close(ProcessId(0)), PoolEffect::Create(ProcessId(1)),]
    );
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Retiring);
    assert_eq!(
        pool.handle(PoolInput::Wake, REFILL_INTERVAL),
        [PoolEffect::WatchExit(ProcessId(0))]
    );
    assert!(pool.handle(PoolInput::Wake, REFILL_INTERVAL * 2).is_empty());
    assert!(
        pool.handle(PoolInput::Closed(ProcessId(0)), REFILL_INTERVAL * 2).is_empty(),
        "close completion cannot stand in for exit or reaping"
    );
    assert!(pool.handle(PoolInput::Exited(ProcessId(0)), REFILL_INTERVAL * 2).is_empty());
    assert_eq!(pool.phase(ProcessId(0)), ProcessPhase::Retiring);
    assert_eq!(
        pool.handle(
            PoolInput::Joined { process: ProcessId(0), outcome: PoolJoinOutcome::Reaped },
            REFILL_INTERVAL * 2
        ),
        [PoolEffect::Released { process: ProcessId(0), outcome: PoolJoinOutcome::Reaped }]
    );
}
