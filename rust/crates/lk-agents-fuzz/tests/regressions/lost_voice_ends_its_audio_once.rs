//! A voice lost after its audio played ends the segment's audio leg once.
use lk_agents_fuzz::drive::{Outcome, Run, Setup, finish, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};
use lk_agents_voice::events::{ErrorSource, SessionEvent};
use lk_agents_voice::trace::{PlayoutStep, Step};

// Catches: a synthesis failing after its audio played reported its segment as heard with no audio,
// so the segment ended its audio leg a second time as an unvoiced one; the synchroniser paired the
// next speech's words with that empty audio and showed them all before any of their audio played
// (`transcript_follows_audio`, campaign seed 20655; a7e9506e9).
#[test]
fn the_next_speech_words_follow_its_own_audio() {
    let text = |t: &str| AgentText(t.into());
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: text("Let me check that.") },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            // The voice fails after its audio reached the speaker, which then plays it all.
            Input::TtsFail { tts: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(5_000) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
            Input::Say {
                text: text("Anything else?"),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(5_000) },
        ] {
            run.apply(&input).await;
        }
        let at = |step: fn(&Step) -> bool| {
            run.record.journal().iter().position(|(_, s)| step(s)).expect("the scenario")
        };
        let played = at(|s| {
            matches!(s, Step::Playout(PlayoutStep::PlaybackFinished { interrupted: false, .. }))
        });
        let lost = at(
            |s| matches!(s, Step::Emitted(SessionEvent::Error(e)) if e.source == ErrorSource::Tts),
        );
        assert!(played < lost, "the scenario: the voice is lost after its audio played whole");
        let outcome = finish(run, Outcome::default()).await;
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
