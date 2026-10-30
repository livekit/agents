//! A reply the server creates as the client's request bound runs out is interrupted, not left
//! holding the server: the timeout's cancel crossed the creation on the wire, and the next
//! `response.create` was refused over it (seed 401 before the hand-set delays; now seed 50).
use lk_agents_abi::RealtimeCommandKind;
use lk_agents_fuzz::drive::{Outcome, Run, finish, simulate_with};
use lk_agents_fuzz::search;
use lk_agents_voice::agent_session::effects::Timer;
use lk_agents_voice::trace::{CoreStep, RealtimeStep, Step, TimerName};

#[test]
fn a_reply_created_as_its_request_timed_out_does_not_hold_the_server() {
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(50));
    assert!(setup.realtime.is_some(), "the case must run a realtime model");
    let case = search::explore(setup, 50, 3_000).case;
    let (outcome, journal) = simulate_with(case.schedule, async {
        let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
        for input in &case.inputs {
            run.apply(input).await;
        }
        let journal: Vec<_> = run.record.journal().to_vec();
        (finish(run, Outcome { case: case.clone(), ..Outcome::default() }).await, journal)
    });
    // The scenario: the request's 10 s bound fires, its cancel goes out, and the server's reply
    // to it lands in that same instant.
    let (fired, at) = journal
        .iter()
        .enumerate()
        .find_map(|(i, (at, s))| {
            matches!(
                s,
                Step::Core(CoreStep::TimerFired(TimerName::Session(Timer::RealtimeRequest(_)), _))
            )
            .then_some((i, at.at_ms))
        })
        .expect("the scenario: a realtime request times out");
    let sent = |kind: RealtimeCommandKind| {
        journal[fired..].iter().position(|(when, s)| {
            when.at_ms == at
                && matches!(s, Step::Realtime(RealtimeStep::Sent { kind: k, .. }) if *k == kind)
        })
    };
    let cancel = sent(RealtimeCommandKind::CancelReply).expect("the scenario: the timeout cancels");
    let interrupt = sent(RealtimeCommandKind::Interrupt).expect("the late reply is interrupted");
    assert!(cancel < interrupt, "the reply landed after the cancel crossed it");
    assert!(
        !outcome.failures.iter().any(|f| f.contract == "realtime_server_accepts_requests"),
        "{:#?}",
        outcome.failures
    );
}
