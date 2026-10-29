//! A reply whose voice is lost before any audio shows its words whole, not paced by the next
//! reply's audio.
use lk_agents_fuzz::drive::{Outcome, Run, Setup, TtsAlignment, finish, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
use lk_agents_fuzz::record::Sink;
use lk_agents_voice::trace::{GenerationStep, Step};

// Catches: with the voice down from the start, the segment's binding was activated only as its
// queued words were forwarded, after the segment had decided against ending its audio leg (no
// transcript chain open yet); the synchroniser kept the words until the next reply's audio paced
// them (`aligned_transcript_follows_tts`, survey seed 27391).
#[test]
fn a_reply_whose_voice_was_down_shows_its_words_before_the_next_reply() {
    let text = |t: &str| AgentText(t.into());
    let setup = Setup {
        sync_transcription: true,
        tts_alignment: TtsAlignment::Words,
        aligned_transcript: Some(true),
        ..Setup::default()
    };
    simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::TtsOutage { down: true },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: text("I'm sorry, I didn't catch that.") },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(10_000) },
            Input::TtsOutage { down: false },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(1), text: text("Done, your refund is on its way.") },
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(5_000) },
        ] {
            run.apply(&input).await;
        }
        // The first reply's words reached the text output before the next reply's synthesis.
        let next = run.record.journal().iter().find_map(|(at, s)| match s {
            Step::Generation(GenerationStep::TtsRequested { .. }) => Some(*at),
            _ => None,
        });
        let next = next.expect("the second reply was synthesised");
        let shown = run.record.sink().iter().any(|(at, call)| {
            *at < next && matches!(call, Sink::Text { text, .. } if text.contains("catch"))
        });
        assert!(shown, "the lost voice's words waited for the next reply's audio");
        let outcome = finish(run, Outcome::default()).await;
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
