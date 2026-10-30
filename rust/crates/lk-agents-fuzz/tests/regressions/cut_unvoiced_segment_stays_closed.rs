//! A speech cut after its segment completed unvoiced does not end that segment's audio again.
use lk_agents_fuzz::drive::{NodeSetup, Outcome, Run, Setup, finish, simulate};
use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput};
use lk_agents_voice::trace::{InterruptStep, PlayoutStep, SinkCall, Step};

// Catches: a segment whose voice ended with no audio was cut by its speech's interruption (the
// cut closed its words), then ended its audio leg as an unvoiced one anyway; the synchroniser
// opened a segment for that audio end and showed the next speech's words whole before any of
// their audio played (`transcript_follows_audio`, campaign seed 19354; 242c57d69).
#[test]
fn the_next_speech_words_wait_for_their_audio() {
    let text = |t: &str| AgentText(t.into());
    // A `tts_node` and no TTS: a reply's words are shown with no audio.
    let setup = Setup {
        tts: false,
        sync_transcription: true,
        nodes: Some(NodeSetup {
            llm: false,
            tts: true,
            stt: false,
            llm_text: None,
            transcription: true,
            realtime_audio: false,
        }),
        ..Setup::default()
    };
    simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: text("Okay. I've moved you to Friday.") },
            Input::Advance { ms: Millis(500) },
            Input::Interrupt { force: true, wait: false },
            Input::Advance { ms: Millis(500) },
            Input::Say {
                text: text("Anything else?"),
                input: SayInput::Audio,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(20) },
            Input::FeedAudio { stream: Ref::new(0), chunks: Count(20) },
            Input::FeedEnd { stream: Ref::new(0) },
            Input::Advance { ms: Millis(5_000) },
        ] {
            run.apply(&input).await;
        }
        let at = |step: fn(&Step) -> bool| {
            run.record.journal().iter().position(|(_, s)| step(s)).expect("the scenario")
        };
        let forwarded = at(|s| matches!(s, Step::Playout(PlayoutStep::TranscriptForwarded { .. })));
        let cut = at(|s| matches!(s, Step::Interrupt(InterruptStep::Interrupted { .. })));
        let pushed = at(|s| {
            matches!(s, Step::Playout(PlayoutStep::SinkCalled { call: SinkCall::Push { .. }, .. }))
        });
        assert!(
            forwarded < cut && cut < pushed,
            "the scenario: words with no audio cut, then audio"
        );
        let outcome = finish(run, Outcome::default()).await;
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
