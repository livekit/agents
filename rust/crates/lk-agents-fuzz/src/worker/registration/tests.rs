use super::*;

/// A run that keeps every rule and reached a reconnect, a launch and a migration.
fn reached_run() -> Trace {
    (0..400)
        .map(|seed| run(seed, Sampling::BasicPos))
        .find(|o| {
            o.violations.is_empty()
                && ["WorkerReconnected", "WorkerJobLaunched", "WorkerJobsMigrated"]
                    .iter()
                    .all(|r| o.reached.contains(r))
        })
        .expect("a seed reconnects with a launched job")
        .trace
}

fn breaks(rule: fn(&Trace) -> Vec<String>, mutate: impl FnOnce(&mut Trace)) {
    let mut trace = reached_run();
    assert!(rule(&trace).is_empty(), "{:?}", rule(&trace));
    mutate(&mut trace);
    assert!(!rule(&trace).is_empty(), "the counterexample passed");
}

fn step_with(trace: &mut Trace, what: impl Fn(&Step) -> bool) -> &mut Step {
    trace.steps.iter_mut().find(|s| what(s)).expect("the trace has such a step")
}

fn launch(s: &Step) -> Option<WorkerStateEffect> {
    s.effects.iter().find(|e| matches!(e, WorkerStateEffect::LaunchJob { .. })).cloned()
}

#[test]
fn a_status_before_register_is_rejected() {
    breaks(registers_first, |t| {
        let s = step_with(t, |s| matches!(s.input, In::Connect));
        let status = lk_agents_worker::protocol::update_worker_draining(0);
        s.effects.insert(0, WorkerStateEffect::Send(status));
    });
}

#[test]
fn an_offer_answered_twice_is_rejected() {
    breaks(answered_once, |t| {
        let s = step_with(t, |s| sent(s).any(|m| matches!(m, Out::Availability(_))));
        let again = s.effects.iter().find(|e| matches!(e, WorkerStateEffect::Send(_))).cloned();
        s.effects.push(again.unwrap());
    });
}

#[test]
fn a_job_launched_twice_is_rejected() {
    breaks(one_disposition, |t| {
        let s = step_with(t, |s| launch(s).is_some());
        let again = launch(s).unwrap();
        s.effects.push(again);
    });
}

#[test]
fn a_job_dropped_before_its_end_or_left_out_of_the_migration_is_rejected() {
    breaks(jobs_kept, |t| {
        let i = t.steps.iter().position(|s| !s.after.active.is_empty()).unwrap();
        let job = t.steps[i].after.active.iter().next().unwrap().clone();
        t.steps[i + 1].after.active.remove(&job);
        // its own final status, if it comes, is later
        for s in &mut t.steps[..=i + 1] {
            if matches!(&s.input, In::Status { job: j, .. } if *j == job) {
                s.input = In::Wake;
            }
        }
    });
    breaks(jobs_kept, |t| {
        let s = step_with(t, |s| sent(s).any(|m| matches!(m, Out::MigrateJob(_))));
        s.effects.retain(|e| !matches!(e, WorkerStateEffect::Send(m) if matches!(m.message, Some(Out::MigrateJob(_)))));
        let empty = lk_agents_worker::protocol::migrate_jobs(vec!["J_none".into()]);
        s.effects.push(WorkerStateEffect::Send(empty));
    });
}

#[test]
fn a_request_left_reserved_is_rejected() {
    breaks(accepted_settle, |t| {
        let last = t.steps.last_mut().unwrap();
        last.after.reserved = 1;
    });
}

#[test]
fn a_drain_ending_twice_or_with_jobs_left_is_rejected() {
    breaks(drain_once, |t| {
        let s = step_with(t, |s| !s.after.active.is_empty());
        s.effects.push(WorkerStateEffect::DrainFinished);
    });
}

#[test]
fn a_seed_replays_its_run() {
    let (a, b) = (run(5, Sampling::BasicPos), run(5, Sampling::BasicPos));
    let inputs =
        |o: &Outcome| o.trace.steps.iter().map(|s| (s.at, s.input.clone())).collect::<Vec<_>>();
    assert_eq!(inputs(&a), inputs(&b));
}
