//! A recognition stream that finishes opening after the session ended is closed.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{Input, Millis, Models};

// Catches: a reopened leg whose connection landed after the loop stopped was dropped with its
// event, never reaching the provider's `close` (`provider_streams_closed`, survey seed 1876).
#[test]
fn a_stream_opened_after_the_close_is_closed() {
    let inputs = vec![
        Input::UpdateModels { models: Models::Stt },
        Input::Advance { ms: Millis(20) },
        Input::Close { drain: false },
    ];
    let setup = Setup { stt: true, connect_ms: Some(15_000), ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"provider_streams_closed"), "{failed:?}");
}
