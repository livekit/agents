//! An interruption may clear only the destination physically admitted for that speech.
use lk_agents_fuzz::search;

#[test]
fn interrupting_a_captured_output_does_not_clear_it_before_physical_admission() {
    let seed = 149;
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
    let outcome = search::explore(setup, seed, 3_000);
    assert!(outcome.expectations.get("output_replacement_is_clean").is_some_and(|n| *n > 0));
    assert!(outcome.case.inputs.iter().any(|input| matches!(
        input,
        lk_agents_fuzz::input::Input::SetAudioOutput { attach: true }
    )));
    assert!(
        outcome
            .case
            .inputs
            .iter()
            .any(|input| matches!(input, lk_agents_fuzz::input::Input::Interrupt { .. }))
    );
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}

#[test]
fn disabling_selected_text_keeps_its_admitted_segment_until_it_finishes() {
    let seed = 5;
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
    let outcome = search::explore(setup, seed, 3_000);
    assert!(outcome.case.inputs.iter().any(|input| matches!(
        input,
        lk_agents_fuzz::input::Input::SetIoEnabled {
            leg: lk_agents_voice::agent_session::IoKind::Transcription,
            enabled: false
        }
    )));
    assert!(outcome.expectations.get("output_replacement_is_clean").is_some_and(|n| *n > 0));
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}
