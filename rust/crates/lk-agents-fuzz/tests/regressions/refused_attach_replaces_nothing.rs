//! An output the session refused to attach replaces nothing.
use lk_agents_fuzz::authored::Authored;
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::input::Input;
use lk_agents_fuzz::record::Sink;

// Catches: `output_replacement_is_clean` reading every output instance the harness built as
// attached. The harness builds the fake before handing it to the session, which refuses it once
// closed; the refused instance then counted as the replacement, and words the current output
// still took read as escaping onto a detached one (survey seed 2597).
#[test]
fn a_refused_attach_replaces_nothing() {
    let recording = Authored::new()
        .output(false)
        .refused(Input::SetTranscriptionOutput { attach: true })
        .output(false)
        .sink(Sink::Text { output: 0, text: "Let me pull up your account.".into() })
        .done();
    let (failures, _) = check_all(&recording);
    let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"output_replacement_is_clean"), "{failures:?}");
}
