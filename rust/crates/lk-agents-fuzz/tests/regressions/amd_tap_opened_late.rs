//! AMD's own recognition stream, landing after the detector stopped reading, is closed.
use lk_agents_fuzz::drive::{AmdAnswer, AmdSetup, Case, Setup, run};
use lk_agents_fuzz::input::{Input, Millis};

fn closes_its_stream(answer: AmdAnswer, before_close: Millis) {
    let setup = Setup {
        connect_ms: Some(15_000),
        amd: Some(AmdSetup { own_stt: true, answer, ..AmdSetup::default() }),
        ..Setup::default()
    };
    let inputs = vec![Input::Advance { ms: before_close }, Input::Close { drain: false }];
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"provider_streams_closed"), "{failed:?}");
}

// Catches: the stream `_run_stt` opened for AMD landing after the session closed (the detector's
// loop gone), dropped with its message and never closed (`provider_streams_closed`, survey seed
// 25): a host stream is not cancelled by a drop.
#[test]
fn a_tap_opened_after_the_close_is_closed() {
    closes_its_stream(AmdAnswer::Track, Millis(20));
}

// The same with the verdict already out (`participant_missing`, the callee's track never came)
// when the stream lands.
#[test]
fn a_tap_opened_after_the_verdict_is_closed() {
    closes_its_stream(AmdAnswer::Missing, Millis(6_000));
}
