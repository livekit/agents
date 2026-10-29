//! A reply the server creates as the client's request bound runs out is interrupted, not left
//! holding the server: seed 401 created it in the instant the 10 s timer fired, the timeout's
//! cancel crossed the creation, and the next `response.create` was refused over it.
use lk_agents_fuzz::search;

#[test]
fn a_reply_created_as_its_request_timed_out_does_not_hold_the_server() {
    let seed = 401;
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
    assert!(setup.realtime.is_some(), "the case must run a realtime model");
    let outcome = search::explore(setup, seed, 3_000);
    assert!(
        !outcome.failures.iter().any(|f| f.contract == "realtime_server_accepts_requests"),
        "{:#?}",
        outcome.failures
    );
}
