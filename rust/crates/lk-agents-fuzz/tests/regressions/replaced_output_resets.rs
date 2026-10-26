//! A replaced output that resets its own playback count is not output reaching it.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref};

// Catches: `output_replacement_is_clean` counting `SinkReset` -- the output throwing its playback
// count away, which the case reports like a playout -- as a call the product made on a replaced
// instance (528 of the rule's failures over seeds 0 to 1023).
#[test]
fn a_reset_on_a_replaced_output_is_not_a_call() {
    let inputs = [
        // Before the driver has swapped it out: the old output is still the case's to reset.
        Input::SetAudioOutput { attach: true },
        Input::SinkReset { output: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    assert!(run.record.applied().iter().all(|a| a.accepted), "the reset was delivered");
    let (failures, _) = check_all(&run.record);
    let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"output_replacement_is_clean"), "{failed:?}");
}
