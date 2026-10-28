//! An output that throws its playback count away reports what it owed as cut first.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};
use lk_agents_voice::events::SessionEvent;

// Catches: the fake output resetting its count without reporting the segment it owed, which no
// sink does (the reference's `DataStreamAudioOutput` reports it finished, then resets). The
// wrapper above it kept counting the segment, so the speech waiting on it, or the next one,
// waited until the close (`speech_settles`, survey seed 755; `synthesized_speech_is_played`).
#[test]
fn a_reset_releases_the_speech_waiting_on_playout() {
    let inputs = [
        Input::Say {
            text: AgentText("Hello there.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(100) },
        Input::SinkReset { output: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
    ];
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let host = run.record.host_view();
    let speeches = host.speeches();
    let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
    assert!(speech.is_done(), "the speech still waits on a playout the output forgot");
}

// Catches: a speech whose open segment the output cut and reset under it, then played the rest,
// committed as complete: the last segment's finish was taken for the whole speech, though the
// caller never heard its first words (survey seeds 24667, 6113, `shown_transcript_is_conserved`).
#[test]
fn a_speech_cut_by_a_reset_is_not_complete() {
    let text = |t: &str| AgentText(t.into());
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmText { chat: Ref::new(0), text: text("Sure, I can") },
        Input::Advance { ms: Millis(20) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::SinkReset { output: Ref::new(0) },
        Input::LlmText { chat: Ref::new(0), text: text(" help with that.") },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::PlayoutStarted { segment: Ref::new(1) },
        Input::PlayoutFinished { segment: Ref::new(1) },
        Input::Advance { ms: Millis(100) },
    ];
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let said: Vec<bool> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::Message(m) if m.role == Role::Assistant => Some(m.interrupted),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(said, [true], "the speech's history entry claims it all played");
}
