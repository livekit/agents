//! A console microphone reaches its listeners through the transport's ordered delivery.
use lk_agents_fuzz::search;

#[test]
fn console_microphone_delivery_retains_transport_order() {
    let seed = 25_066;
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
    assert!(setup.console, "the case must exercise the console transport");
    let outcome = search::explore(setup, seed, 3_000);
    assert!(outcome.expectations.get("forwarded_while_enabled").is_some_and(|n| *n > 0));
    assert!(
        !outcome.failures.iter().any(|failure| failure.contract == "forwarded_while_enabled"),
        "{:#?}",
        outcome.failures
    );
}
