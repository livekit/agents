//! A reply the server creates as the client's request bound runs out is interrupted, not left
//! holding the server.
use lk_agents_abi::RealtimeCommandKind;
use lk_agents_fuzz::drive::{Outcome, RealtimeSetup, Run, Setup, finish, simulate};
use lk_agents_fuzz::fakes::RealtimeDelays;
use lk_agents_fuzz::input::{Input, Millis};
use lk_agents_voice::agent_session::effects::Timer;
use lk_agents_voice::trace::{CoreStep, RealtimeStep, Step, TimerName};

// Catches: the timeout's cancel crossing the server's creation on the wire. The late generation
// was discarded without an interrupt and held the server, which refused the next
// `response.create` (survey seeds 401 and 50; eca089161).
#[test]
fn a_reply_created_as_its_request_timed_out_does_not_hold_the_server() {
    // The server answers exactly at the request's 10 s bound.
    let delays = RealtimeDelays { created: Some(10_000), ..RealtimeDelays::default() };
    let setup = Setup {
        realtime: Some(RealtimeSetup { turn_detection: false, delays, ..RealtimeSetup::default() }),
        ..Setup::default()
    };
    let reply = || Input::GenerateReply { user_input: None, instructions: None, allow: None };
    let inputs = [reply(), Input::Advance { ms: Millis(10_100) }, reply()];
    let (outcome, journal) = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        let journal: Vec<_> = run.record.journal().to_vec();
        (finish(run, Outcome::default()).await, journal)
    });
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
        .expect("the scenario: the request times out");
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
