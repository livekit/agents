//! Audio synthesised before the voice failed plays out before the speech ends.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};

// Catches: a TTS failing after its first chunk left that chunk playing unwaited, so the speech
// completed before any of its audio was heard (`synthesized_speech_is_played`, survey seed 122).
// The reference's `forward_generation` waits for the playout whether or not the TTS failed.
#[test]
fn a_failed_synthesis_waits_for_what_it_made() {
    let text = |t: &str| AgentText(t.into());
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        let opening = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: text("Hello there.") },
            Input::Advance { ms: Millis(20) },
            Input::TtsHold { tts: Ref::new(0) },
            Input::LlmText { chat: Ref::new(0), text: text(" How are you?") },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::TtsFail { tts: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
        ];
        for input in &opening {
            run.apply(input).await;
        }
        assert!(run.record.applied().iter().all(|a| a.accepted), "every input landed");
        let speeches = run.record.host_view().speeches();
        let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
        assert!(!speech.is_done(), "the speech ended before its audio played");
        for input in
            [Input::PlayoutFinished { segment: Ref::new(0) }, Input::Advance { ms: Millis(20) }]
        {
            run.apply(&input).await;
        }
        run
    });
    let speeches = run.record.host_view().speeches();
    assert!(speeches[0].completed_normally(), "the playout released the speech");
}
