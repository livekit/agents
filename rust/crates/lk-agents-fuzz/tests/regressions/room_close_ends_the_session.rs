//! The room closing ends the session, however the room IO's own close races it.
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_fuzz::drive::{Outcome, Run, finish, simulate_with};
use lk_agents_fuzz::input::{Input, RoomIn};
use lk_agents_fuzz::search;
use lk_agents_voice::trace::{CloseStep, RoomEventKind, RoomStep, Step};

// Catches: the room IO's `Close` stopped its forwarder before the forwarder passed the same
// `Close` (and the link's end) to the session, so the session never closed and an IVR reply
// later played into the closed room output, cut with no cause (`speech_is_heard`, campaign seed
// 19655).
#[test]
fn the_session_closes_with_its_room() {
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(19_655));
    let case = search::explore(setup, 19_655, 3_000).case;
    assert!(
        case.inputs.iter().any(|i| matches!(i, Input::Room { event: RoomIn::Closed })),
        "the scenario: the room closes during the call"
    );
    let (outcome, journal) = simulate_with(case.schedule, async {
        let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
        for input in &case.inputs {
            run.apply(input).await;
        }
        let journal: Vec<Step> = run.record.journal().iter().map(|(_, s)| s.clone()).collect();
        (finish(run, Outcome { case: case.clone(), ..Outcome::default() }).await, journal)
    });
    let closed = journal
        .iter()
        .position(|s| matches!(s, Step::Room(RoomStep::Event(RoomEventKind::Close))))
        .expect("the session hears the room close");
    assert!(
        journal[closed..].iter().any(|s| matches!(s,
            Step::Close(CloseStep::Requested { reason: CloseReason::JobShutdown, .. }))),
        "the room's close ends the session"
    );
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}
