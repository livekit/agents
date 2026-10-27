//! AMD's own recognition stream, opened while the detector was closing, is closed.
use lk_agents_fuzz::drive::{AmdAnswer, AmdSetup, Case, Setup, run};
use lk_agents_fuzz::input::{AmdInput, Input, Millis};

// Catches: the stream `_run_stt` opened for AMD queued on the detector's channel as `AMD.aclose()`
// ended its loop: dropped with the channel, never handled and never closed
// (`provider_streams_closed`, survey seed 7022): a host stream is not cancelled by a drop.
#[test]
fn a_tap_queued_as_the_detector_closes_is_closed() {
    let setup = Setup {
        amd: Some(AmdSetup { own_stt: true, answer: AmdAnswer::Track, ..AmdSetup::default() }),
        ..Setup::default()
    };
    let inputs = vec![
        Input::Amd { input: AmdInput::Close },
        Input::Advance { ms: Millis(100) },
        Input::Close { drain: false },
    ];
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"provider_streams_closed"), "{failed:?}");
}
