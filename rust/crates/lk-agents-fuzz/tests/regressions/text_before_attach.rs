//! Words reach a transcription output only once its binding attached it.
use lk_agents_fuzz::drive::{Outcome, Run, Setup, finish, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
use lk_agents_fuzz::record::Sink;

// Catches: an authorized synthesised segment forwarding its words straight to the transcription
// output it captured, ahead of the activation that attaches it. With the voice down until the
// output is replaced, the words written as the synthesis gives up reach an output that was never
// attached (`output_replacement_is_clean`, survey seed 1560 and 17 more in the 32768 window). The
// segment activates after the replacement, so all its words go to the replacement.
#[test]
fn a_failed_voice_writes_no_words_to_an_unattached_replaced_output() {
    let text = |t: &str| AgentText(t.into());
    simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in [
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
        ] {
            run.apply(&input).await;
        }
        // Output 1 is the replaced transcription output, output 2 its replacement.
        let text: Vec<_> = run
            .record
            .sink()
            .iter()
            .filter_map(|(_, call)| match call {
                Sink::Text { output, text } => Some((*output, text.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(text, [(2, "Hello there."), (2, " How are you?")]);
        let outcome = finish(run, Outcome::default()).await;
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
