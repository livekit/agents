//! A `say` the close's edge released before its step reported is still committed as a `say`.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput, UserText};
use lk_agents_voice::agent::Source;
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::Step;

// Catches: the item of a `say` cut by the close, whose step reported spoken only after the close's
// hard edge had released the speech, read the speech's kind off the scheduler's entry, gone with
// the release: it was committed as a reply's item (`Source::Reply`), and the host was told a
// pipeline reply had been voiced without naming its voice (`items_carry_their_payload`, survey
// seed 114526).
#[test]
fn a_say_released_by_the_close_keeps_its_source() {
    let setup = Setup { tts: true, stt: true, ..Setup::default() };
    let inputs = [
        Input::Say {
            text: AgentText(String::new()),
            input: SayInput::TextStream,
            allow: None,
            to_ctx: true,
        },
        Input::FeedText { stream: Ref::new(0), text: AgentText("One moment please.".into()) },
        Input::Advance { ms: Millis(2_000) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        // The person is mid-sentence: the close waits for their transcript past its edge.
        Input::SttStart { stt: Ref::new(0) },
        Input::SttInterim { utterance: Ref::new(0), text: UserText("wait I".into()), speaker: None },
        // The caller's text stream is still open as the session closes: the step cannot end
        // before the edge releases the speech.
        Input::Close { drain: true },
        // Just past the edge, which released the speech, inside the transcript wait.
        Input::Advance { ms: Millis(10_050) },
        Input::FeedEnd { stream: Ref::new(0) },
        Input::Advance { ms: Millis(5_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let sources: Vec<&Source> = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, step)| match step {
            Step::Emitted(SessionEvent::ConversationItemAdded { item, source, .. }) => {
                matches!(&**item, ChatItem::Message(m) if m.role == Role::Assistant).then_some(source)
            }
            _ => None,
        })
        .collect();
    assert!(!sources.is_empty(), "the say's words were committed");
    assert!(
        sources.iter().all(|s| matches!(s, Source::Say { .. })),
        "the say's item keeps its source: {sources:?}"
    );
}
