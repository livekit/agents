//! Speeches queued behind `min_consecutive_speech_delay` all play out in the release phase.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};

// Catches: the release phase ending on two rounds with nothing to release while the driver still
// held a speech behind its own consecutive-speech timer, which `speech_settles` then judged stuck
// (survey seeds 34, 172 and 412, among nine).
#[test]
fn speeches_behind_the_gap_settle() {
    let say = |text: &str| Input::Say {
        text: AgentText(text.into()),
        input: SayInput::Words,
        allow: None,
        to_ctx: true,
    };
    let mut inputs: Vec<Input> = ["One.", "Two.", "Three.", "Four."].map(say).into();
    inputs.push(Input::Advance { ms: Millis(20) });
    let setup = Setup { speech_gap_ms: Some(15_000), ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"speech_settles"), "{failed:?}");
    assert!(!outcome.stopped_early);
}
