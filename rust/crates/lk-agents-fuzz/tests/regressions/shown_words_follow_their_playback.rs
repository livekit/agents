//! A shown word is the speech's whose playback released it, and counts against that playback.
use lk_agents_fuzz::drive::{Outcome, Run, Setup, finish, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
use lk_agents_voice::trace::{PlayoutStep, Step};

// Catches: after a sink reset the rest of a speech's audio plays as a segment of its own, whose
// reports the speech's playout dropped as another key's: its played position stayed where the
// reset left it, and the words that playback released were journalled against it, ahead of the
// audio (`transcript_follows_audio`, campaign seed 13704; d5f0e2ff9).
#[test]
fn the_rest_after_a_sink_reset_is_the_speechs_playback() {
    let text = |t: &str| AgentText(t.into());
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: text("Your order ships on Tuesday") },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(500) },
            // The sink resets mid-speech: the rest of its audio plays as a segment of its own.
            Input::SinkReset { output: Ref::new(0) },
            Input::LlmText { chat: Ref::new(0), text: text(" and arrives by Friday afternoon.") },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutStarted { segment: Ref::new(1) },
            Input::Advance { ms: Millis(1_000) },
            Input::PlayoutProgressed { segment: Ref::new(1), ms: Millis(5_000) },
            Input::PlayoutFinished { segment: Ref::new(1) },
            Input::Advance { ms: Millis(1_000) },
        ] {
            run.apply(&input).await;
        }
        let journal: Vec<&Step> = run.record.journal().iter().map(|(_, s)| s).collect();
        let reset = journal
            .iter()
            .position(|s| matches!(s, Step::Playout(PlayoutStep::SinkReset { .. })))
            .expect("the scenario: a sink reset mid-speech");
        assert!(
            journal[reset..]
                .iter()
                .any(|s| matches!(s, Step::Playout(PlayoutStep::TranscriptShown { .. }))),
            "the rest's playback released none of the speech's words"
        );
        let outcome = finish(run, Outcome::default()).await;
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
