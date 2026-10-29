//! Words reach a transcription output only once its binding attached it.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};

// Catches: an authorized synthesised segment forwarding its words straight to the transcription
// output it captured, ahead of the activation that attaches it. With the voice down until the
// output is replaced, the words written as the synthesis gives up reach an output that was never
// attached (`output_replacement_is_clean`, survey seed 1560 and 17 more in the 32768 window).
#[test]
fn a_failed_voice_writes_no_words_to_an_unattached_replaced_output() {
    let text = |t: &str| AgentText(t.into());
    let inputs = vec![
        Input::TtsOutage { down: true },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmText { chat: Ref::new(0), text: text("Hello there.") },
        Input::Advance { ms: Millis(20) },
        Input::SetTranscriptionOutput { attach: true },
        Input::Advance { ms: Millis(20) },
        Input::LlmText { chat: Ref::new(0), text: text(" How are you?") },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::Advance { ms: Millis(10_000) },
    ];
    let outcome = run(Case { setup: Setup::default(), inputs, ..Case::default() });
    assert!(outcome.expectations.get("output_replacement_is_clean").is_some_and(|n| *n > 0));
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}
